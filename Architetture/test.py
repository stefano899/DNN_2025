import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    accuracy = accuracy_score(all_labels, all_preds)

    ## FOR HAVING MORE PRECISION ON RESULTS
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Error: \n Accuracy: {accuracy:>8f}%, Avg loss: {test_loss:>8f}%") # for more results, delete this parethesys and add f" \n F1_Score: {f1:>8f}% " f"\n Precision: {precision:>8f}% \n Recall: {recall:>8f}%")

    return accuracy, test_loss #, f1, precision, recall
