# Source: radixark/miles 41c5e38b94ea23677de93b01a4a77d55677a8f09
# Image: radixark/miles:dev-202609251434
# miles/backends/megatron_utils/actor.py (unmodified relevant functions; imports/decorators omitted in excerpts)
class PinnedMethods:
    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        rollout_id: int,
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                rollout_id=rollout_id,
                store_prefix=store_prefix,
                fp32_output=False,
                use_rollout_sampling_mask=store_prefix == "" and self.args.use_sampling_support_replay,
            )

    def train(
        self,
        rollout_id: int,
        rollout_data_ref: Box,
        witness_info: WitnessInfo | None = None,
        attempt: int = 0,
        external_data=None,
    ):
        self._heartbeat.bump()
        self._last_rollout_id = rollout_id
        if self.args.offload_train and self._asleep:
            self.wake_up()

        with ExitStack() as stack:
            with timer("data_preprocess"):
                rollout_data, store_get_result = get_rollout_data(
                    self.args, rollout_data_ref, witness_info=witness_info
                )
                stack.enter_context(store_get_result)
                if self.args.debug_rollout_only:
                    log_rollout_data(rollout_id, self.args, rollout_data)
                    return TrainStepOutput(outcome=TrainStepOutcome.NORMAL)

            if self.role == "critic":
                with timer("critic_train"):
                    result = self.train_critic(rollout_id, rollout_data)
            else:
                result = self.train_actor(
                    rollout_id,
                    rollout_data,
                    external_data=external_data,
                    witness_info=witness_info,
                    attempt=attempt,
                )

            return result
