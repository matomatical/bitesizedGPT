import sys
import time
import tqdm
import torch
from model import ByteTransformer, ByteCorpus
from model import complete, next_byte_cross_entropy_loss


def train():
    print("choosing device...")
    if sys.argv[1:]:
        DEVICE = sys.argv[1]
    else:
        DEVICE = 'cpu'
    print("device:", DEVICE)

    print("loading data...")
    data = ByteCorpus(
        path="data/sherlock-ascii.txt",
        device=DEVICE,
    )

    print("initialising model...")
    model = ByteTransformer(
        max_context_length=128,
        embed_size=32,
        mlp_size=64,
        num_heads=4,
        num_layers=4,
        device=DEVICE,
    )
    model.train()
    
    # initialising training stuff
    num_training_steps = 50000
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    print("training model...")
    for steps in tqdm.trange(num_training_steps):
        batch = data.get_training_batch(seq_length=128, batch_size=32)
        logits = model(batch)
        loss = next_byte_cross_entropy_loss(batch, logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # evaluate periodically
        if steps % 100 == 0:
            tqdm.tqdm.write(f"eval at step {steps:>6d}")
            model.eval()
            # batch loss
            tqdm.tqdm.write(f"  training loss: {loss.item():>6.3f}")
            # test loss
            eval_batch = data.get_testing_batch(seq_length=128, batch_size=256)
            with torch.no_grad():
                logits = model(eval_batch)
            loss = next_byte_cross_entropy_loss(eval_batch, logits)
            tqdm.tqdm.write(f"  testing loss:  {loss.item():>6.3f}")
            # prompt
            with torch.no_grad():
                ctn = complete(model, '"Elementary, my dear', max_bytes=32)
            tqdm.tqdm.write(f"  continuation:  {ctn!r}")
            model.train()

    print("done!")
    print("generating passage from model...")
    model.to('cpu')
    ctn = complete(model, '"Elementary, my dear', max_bytes=512)
    for c in ctn:
        print(c, end="", flush=True)
        time.sleep(0.05)


if __name__ == "__main__":
    train()
