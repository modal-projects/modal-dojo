# Source: radixark/miles 41c5e38b94ea23677de93b01a4a77d55677a8f09
# Image: radixark/miles:dev-202609251434
# miles/ray/train/group.py (unmodified relevant functions; imports/decorators omitted in excerpts)
class PinnedMethods:
    async def update_weights(self, rollout_id: int | None = None) -> int | None:
        """Broadcast weights to rollout engines and answer the version they now serve."""
        log_structured(logger.info, tag="ft", op="update_weights", phase="start", rollout=rollout_id)
        # TODO: allow using all cells to update weights (instead of first alive cell)
        # Fetch the updatable engines once (like V1 RayActorGroup) so all
        # ranks observe a consistent engine set.
        info = await self._inference_controller.start_update_weights()
        # Catch with vanilla retry: cells w/ exceptions are auto marked errored, thus retry will find the next one
        weight_versions = await retry(
            lambda _: self._execute_first_alive("update_weights", info=info),
            max_attempts=_RETRY_MAX_ATTEMPTS,
        )
        await self._inference_controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

        await self._maybe_log_inference_engine_weight_checksums(rollout_id=rollout_id)

        return weight_versions[0]
