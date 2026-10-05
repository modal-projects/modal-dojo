# 02 — Entropy behaves unexpectedly, or performance collapses

Token entropy measures the spread of next-token probabilities at a given prefix. Read it alongside task performance and response diversity: its direction alone does not identify a problem.

## Inspect measurement and responses

**First confirm entropy is being computed.** Logging entropy does not require an entropy bonus in the loss; zero or constant values may be logging placeholders.

Average entropy over the token positions included in the policy-gradient loss, using the same token mask. Inspect the per-token entropy distribution early to establish a baseline, then follow the trend with consistent masking and averaging.

**Read the pattern.**

- **A — Performance improves as entropy falls.** The policy may be concentrating on useful behavior. Lower entropy alone is not a reason to intervene.
- **B — Entropy falls sharply and progress stalls.** This can indicate **entropy collapse**: the policy becomes increasingly deterministic, restricting exploration. Inspect repeated attempts at the same prompts and check sampling settings.
- **C — Entropy rises while quality falls.** Inspect repetition, incoherence and degraded reasoning, then compare against a control for the suspected setting. The example below follows this investigation.

**Before choosing a fix:** confirm consistent entropy measurement, inspect responses and held-out performance, then investigate changes in the objective, updates or data. The same entropy pattern can have different causes.

## Choose an intervention, then verify learning

**Rising entropy with worsening performance does not uniquely diagnose an excessive entropy bonus.** Reducing or removing it is one controlled test when it is enabled. Without a bonus, investigate the rest of the training loop:

| Evidence to check | Next action |
| --- | --- |
| Abrupt policy changes: learning rate, probability ratios, clipping and gradients | If updates are excessive, test a lower learning rate or fewer updates per collected batch ([04](debug-updates.md)). |
| Trainer–rollout disagreement at matching weights and prefixes | Check scoring conventions, inputs and weight loading; repair the identified mismatch ([03](debug-logprobs.md)). |
| Changed task mix, unreliable rewards or incorrect advantages | Inspect traces, verifier decisions and batch construction. Fix the data or signal; recheck on a fixed evaluation set ([01](debug-reward.md) and [04](debug-updates.md)). |

For **falling entropy with stalled learning**, check unintended deterministic sampling first. If updates concentrate behavior too quickly, test their size before adding an entropy bonus.

**Top-p masking is another option to test for entropy preservation and exploration.** Training normalizes probabilities over the token set retained during rollout. For that fixed set, the policy-gradient term has no direct gradient to excluded logits. Shared parameters still change, so entropy preservation is not guaranteed. [Sampling-support implementation](https://github.com/radixark/miles/pull/2596)

**Verify:** compare full-vocabulary entropy, response diversity and held-out performance using consistent scoring and averaging. Entropy after sampling filters is a different quantity. Task mix, prefixes and response lengths can move the mean; judge an intervention by learning and response quality, not by reaching a target entropy.

Reported cases in DAPO: [entropy collapse, with a paired accuracy comparison (§3.1, Figure 2)](https://arxiv.org/html/2503.14476v2#S2.F2), and [excessive entropy and response-length growth (§3.3, Figure 4)](https://arxiv.org/html/2503.14476v2#S3.F4).

## Observed example: rising entropy with falling accuracy

An entropy bonus is intended to discourage premature concentration and preserve exploration. For a minimized loss, it adds a term $-\beta H$ to the RL loss: increasing entropy improves that term regardless of whether the additional token alternatives help solve the task. It can therefore compete with reward-driven learning. [Entropy regularization experiments (§4.1)](https://arxiv.org/html/2505.22617v1#S4.SS1)

Both runs use synchronous GLM-4.7-Flash training on DAPO-Math-17k with an 8K response limit. Evaluation uses the same 1,000 held-out questions, eight responses each and the same 8K limit. The configurations differ only in entropy coefficient and run names.

**1. Locate the deterioration.** With coefficient 0.001, held-out pass@1 falls from 65.81% at update 70 to 29.85% at update 140 while entropy rises sharply. Without the bonus, entropy stays near its initial level and accuracy improves. This is pattern C in the affected run: performance collapses while entropy rises.

**2. Inspect the responses.** In the [affected run's dashboard](https://modal-labs-nan-dev--training-gym-dashboard-fastapi-app.modal.run/training/gravitational-conduit-e90cee961cff), compare earlier rollouts with rollout 139, used for update 140. Here is the unchanged opening of one response to a problem asking for the smallest integer whose digits multiply to $9!$:

> We should find the minimal integer. A standard way: as short a representation of $9!$ as possible, meaning it wants a rounded minimal count of numbers: ergy as partas that must produce $5$. Use algorithm.StatusOK sum factor and Sam...
>

The response contains incoherent text, received reward 0 and used 3,879 tokens, below the 8K cap. This selected example illustrates degraded generation; it does not measure how often it occurs.

**3. Test the hypothesis against the control.** The otherwise matched [run without an entropy bonus](https://modal-labs-nan-dev--training-gym-dashboard-fastapi-app.modal.run/training/snowy-skin-5ba2d9b61bd5) reaches 72.15% held-out pass@1 at update 140 and avoids the same deterioration. This supports omitting the 0.001 bonus in this recipe. It does not establish a universal coefficient or demonstrate recovery of an already degraded policy.
