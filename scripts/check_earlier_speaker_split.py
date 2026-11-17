"""Offline checks for independent speaker streams and the existing training pipeline."""

import torch

from scripts.check_afrcnn import main as pipeline_checks
from src.loss import pit_loss
from src.model import build_model
from src.settings import load_config, seed_all


def main():
    settings = load_config("smoke.yaml")
    torch.set_num_threads(2)
    seed_all(settings["seed"])
    for sources in (2, 3):
        options = {**settings["earlier_speaker_split"], "sources": sources}
        for split in (0, options["iterations"], 1.5):
            try:
                build_model({**options, "split_iteration": split})
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid split_iteration accepted")
        model = build_model(options).eval()
        mixture = torch.randn(2, 257)
        batch_sizes = []
        hook = model.block.register_forward_pre_hook(
            lambda module, args: batch_sizes.append(args[0].shape[0])
        )
        with torch.inference_mode():
            expected = model(mixture)
        hook.remove()
        split = options["split_iteration"]
        assert batch_sizes == [2] * split + [2 * sources] * (options["iterations"] - split)

        def reverse_sources(module, args, output):
            return output.reshape(2, sources, *output.shape[1:]).flip(1).reshape_as(output)

        hook = model.speaker_norm.register_forward_hook(reverse_sources)
        with torch.inference_mode():
            permuted = model(mixture)
        hook.remove()
        torch.testing.assert_close(permuted, expected.flip(1), atol=1e-6, rtol=1e-5)

        # Check activation recomputation produces the same loss and gradients.
        reference = build_model(options).eval()
        reference.load_state_dict(model.state_dict())
        model.train()
        target = torch.randn(2, sources, 257)
        actual_loss = pit_loss(model(mixture), target)
        reference_loss = pit_loss(reference(mixture), target)
        actual_loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(actual_loss, reference_loss)
        for parameter, other in zip(model.parameters(), reference.parameters()):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-5, rtol=1e-4)
        assert model.speaker_split[0].weight.grad.abs().sum() > 0
        print(f"{sources} sources: split placement, permutation equivariance, "
              "and checkpointed gradients passed", flush=True)
    pipeline_checks("earlier_speaker_split")


if __name__ == "__main__":
    main()
