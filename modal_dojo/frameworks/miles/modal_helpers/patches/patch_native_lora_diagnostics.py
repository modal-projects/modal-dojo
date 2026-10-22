"""Pinned Miles integration for before/after update diagnostics."""

from pathlib import Path


def apply(root=Path("/root/miles/miles")):
    root = Path(root)
    actor = root / "backends/megatron_utils/actor.py"
    text = actor.read_text()
    native = "def _compute_log_prob(" in text
    marker = (
        "from modal_dojo.frameworks.miles import lora_diagnostics as update_diagnostics"
    )
    if marker not in text:
        anchor = '                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")\n'
        if text.count(anchor) != 1:
            raise ValueError("Miles actor probe anchor changed")
        text = text.replace(
            anchor,
            anchor
            + """                update_probe = update_diagnostics.begin(self.args, rollout_data, rollout_id)
                if update_probe is not None and num_optimizer_steps != 1:
                    raise ValueError("Update diagnostics require one optimizer update per batch")
""",
        )
        anchor = "        skip_actor_forward_only = self.args.skip_actor_forward_only\n"
        if text.count(anchor) != 1:
            raise ValueError("Miles actor entry anchor changed")
        text = text.replace(
            anchor,
            "        from modal_dojo.frameworks.miles import lora_diagnostics as update_diagnostics\n        update_probe = None\n"
            + anchor,
        )
        anchor = '            self._set_replay_stage("replay_backward")\n'
        if text.count(anchor) != 1:
            raise ValueError("Miles training anchor changed")
        text = text.replace(
            anchor,
            """            if update_probe is not None:
                update_probe.snapshot_weights(self.model)
"""
            + anchor,
        )
        anchor = "            self.prof.step(rollout_id=rollout_id)\n"
        if text.count(anchor) != 1:
            raise ValueError("Miles post-update anchor changed")
        # The existing forward clears its replay cursors. Refill from the saved
        # rollout routes so both distributions use identical route decisions.
        post_update = """            if update_probe is not None and train_step_outcome == TrainStepOutcome.NORMAL:
                for m in all_replay_managers:
                    if m.enabled:
                        if not self._use_rollout_replay(m):
                            raise ValueError("Update probes require saved rollout routing for enabled replay")
                        m.clear_all()
                        fill_replay_data(
                            args=self.args, models=self.model, data_iterator=data_iterator,
                            num_microbatches=num_microbatches, rollout_data=rollout_data,
                            data_key=m.data_key, replay_list=m.replays,
                            register_replay_list_func=m.register_replay_list_func,
                            if_sp_region=m.if_sp_region,
                            indices_are_token_positions=m.replay_indices_are_token_positions,
                        )
                self._set_replay_stage("replay_forward")
                update_probe.phase = "after"
                self.compute_log_prob(data_iterator, num_microbatches, rollout_id, store_prefix="diagnostic_")
                update_probe.finish(self.model)
            update_diagnostics.end()
"""
        if native:
            post_update = post_update.replace(
                "self.compute_log_prob(", "self._compute_log_prob("
            )
            post_update = post_update.replace(
                "                            indices_are_token_positions=m.replay_indices_are_token_positions,\n",
                "                            indices_are_token_positions=m.replay_indices_are_token_positions,\n"
                "                            sequence_first=m is routing_replay_manager,\n",
            )
        text = text.replace(anchor, post_update + anchor)
        compile(text, str(actor), "exec")
        actor.write_text(text)
    text = actor.read_text()
    retain_marker = (
        "        diagnostic_replay_data = update_diagnostics.retain_replay_data(\n"
    )
    if retain_marker not in text:
        anchor = "        update_probe = None\n"
        if text.count(anchor) != 1:
            raise ValueError("Miles replay retention anchor changed")
        text = text.replace(
            anchor,
            anchor
            + retain_marker
            + "            rollout_data, [m.data_key for m in all_replay_managers if self._use_rollout_replay(m)]\n        )\n",
        )
        anchor = "                        m.clear_all()\n                        fill_replay_data(\n"
        if text.count(anchor) != 1:
            raise ValueError("Miles replay refill anchor changed")
        text = text.replace(
            anchor,
            "                        m.clear_all()\n                        rollout_data[m.data_key] = diagnostic_replay_data.pop(m.data_key)\n                        fill_replay_data(\n",
        )
    init_marker = "        validate_layout(self.model)\n"
    if init_marker not in text:
        anchor = "        # empty cache after initialization\n"
        if text.count(anchor) != 1:
            raise ValueError("Miles initialization probe anchor changed")
        text = text.replace(
            anchor,
            "        from modal_dojo.frameworks.miles.lora_diagnostics import validate_layout\n"
            + init_marker
            + "\n"
            + anchor,
        )
    compile(text, str(actor), "exec")
    actor.write_text(text)
    model = root / "backends/megatron_utils/model.py"
    text = model.read_text()
    marker = "        from modal_dojo.frameworks.miles.lora_diagnostics import capture"
    if marker not in text:
        anchor = (
            "        return output_tensor, partial(f, **callback_kwargs)\n"
            if native
            else "        return output_tensor, partial(\n            f,\n"
        )
        if text.count(anchor) != 1:
            raise ValueError("Miles forward callback anchor changed")
        text = text.replace(
            anchor, marker + "\n        capture(output_tensor, batch)\n\n" + anchor
        )
        compile(text, str(model), "exec")
        model.write_text(text)


if __name__ == "__main__":
    apply()
