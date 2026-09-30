# Source: radixark/miles 41c5e38b94ea23677de93b01a4a77d55677a8f09
# Image: radixark/miles:dev-202609251434
# miles/backends/megatron_utils/model.py (unmodified relevant functions; imports/decorators omitted in excerpts)
def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: OptimizerParamScheduler | None,
    num_microbatches: int,
    num_rollouts: int,
    witness_info: WitnessInfo | None,
    attempt: int,
    ft_test_action_executor: FTTestActionActorExecutor | None = None,
) -> tuple[dict[str, float], float, TrainStepOutcome]:
    """Run pipeline forward/backward, then step the optimizer and scheduler when gradients are valid."""
    args = get_args()
    parallel_state = get_parallel_state()
    dumper_phase_util = DumperMegatronUtil(args, model, DumperPhase.FWD_BWD, rollout_id=rollout_id)
    disable_optimizer = args.debug_disable_optimizer or optimizer is None
    _zero_grads(model, optimizer, disable_optimizer)

    if args.custom_megatron_before_train_step_hook_path:
        from miles.utils.function_registry import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    losses_reduced = run_forward_backward_pass(
        args, dumper_phase_util, data_iterator, model, num_microbatches, num_rollouts
    )

    outcome = TrainStepOutcome.NORMAL
    grad_norm = 0.0
    valid_step = True
    indep_dp_loss_reduced: dict[str, float] = {}

    if parallel_state.indep_dp.size > 1:
        assert step_id == 0, "indep-dp does not support multi step per train yet"

        if ft_test_action_executor is not None:
            ft_test_action_executor.maybe_crash(rollout_id=rollout_id, attempt=attempt)

        metric_num_rollouts = None if args.calculate_per_token_loss else num_rollouts
        ok, indep_dp_loss_reduced = allreduce_grads_and_losses_across_replicas(
            args, model, parallel_state, losses_reduced=losses_reduced, num_rollouts=metric_num_rollouts
        )
        if not ok:
            outcome = TrainStepOutcome.DISCARDED_SHOULD_RETRY
            valid_step = False

    if (not disable_optimizer) and (not getattr(args, "check_for_nan_in_loss_and_grad", True)):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    if args.ci_test and args.enable_mtp_training and args.rollout_max_response_len <= 128:
        # under response length <= 128, all outputs are truncated and loss mask is all zeros, so only MTP parameters have non-zero gradients
        from miles.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    # Dump backward tensors while gradients are still attached. The optimizer
    # step and subsequent zero_grad release them.
    if outcome == TrainStepOutcome.NORMAL:
        dumper_phase_util.finalize(model)

    if not disable_optimizer and valid_step:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        assert update_successful
        opt_param_scheduler.step(increment=num_rollouts)

    _zero_grads(model, optimizer, disable_optimizer)

    log_structured(
        logger.info,
        tag="train",
        op="train_step",
        rollout=rollout_id,
        step=step_id,
        attempt=attempt,
        outcome=outcome.name,
        valid_step=valid_step,
    )

    if outcome == TrainStepOutcome.NORMAL:
        dump_local_weight_checksums(args=args, model=model, optimizer=optimizer)
        if args.enable_witness:
            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            metric_num_rollouts = None if args.calculate_per_token_loss else num_rollouts
            loss_reduced = (
                indep_dp_loss_reduced
                if parallel_state.indep_dp.size > 1
                else aggregate_train_losses(losses_reduced, metric_num_rollouts)
            )
            return loss_reduced, grad_norm, outcome

    return {}, grad_norm, outcome
