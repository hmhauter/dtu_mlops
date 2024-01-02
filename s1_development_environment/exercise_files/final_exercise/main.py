import click
import torch
from torch import nn, optim
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set, _ = mnist()

    epochs = 10

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

    torch.save(model, 'trained_model.pt')



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()

    equals = 0
    total = 0
    model.eval()  
    with torch.no_grad():
        for images, labels in test_set: 
            outputs = torch.exp(model(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            equals += (predicted == labels).sum().item()
    print(f"Accuracy: {(equals/total)*100}%")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
