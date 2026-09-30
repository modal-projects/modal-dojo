# Source: radixark/miles 41c5e38b94ea23677de93b01a4a77d55677a8f09
# Image: radixark/miles:dev-202609251434
# miles/ray/rollout/rollout_executor.py (unmodified relevant functions; imports/decorators omitted in excerpts)
class PinnedMethods:
    async def get(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self._rollouts_since_weight_version_publish += 1
        assert_weight_version_is_published(
            self.args, rollouts_since_publish=self._rollouts_since_weight_version_publish
        )
        if (get_buffer_length := getattr(self.data_source, "get_buffer_length", None)) is not None:
            dashboard_hooks.report_data_buffer(get_buffer_length())
        with timer("rollout"):
            data, metadata, metrics = await self._get_rollout_data(rollout_id=rollout_id)
        save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=False, metadata=metadata)
        log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        data = convert_samples_to_train_data(
            self.args,
            data,
            metadata=metadata,
            custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,
            custom_reward_post_process_func=self.custom_reward_post_process_func,
        )
        sample_indices = data.get("sample_indices")
        if self.args.delay_split_train_data_by_dp:
            data_ref = object_store.get_instance().put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        else:
            data_ref = split_train_data_by_dp(self.args, data, self.train_parallel_config)
        return dict(sample_indices=sample_indices, data_ref=data_ref)
