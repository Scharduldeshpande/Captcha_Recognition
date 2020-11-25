from tqdm import tqdm
import config
import matplotlib.pyplot as plt


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds

def plot_performance(epochs, accuracy, training_loss, test_loss, result_path):
	fig, (ax0, ax1, ax2) = plt.subplots(3)
	fig.suptitle("Model performamce")

	ax0.plot(epochs, training_loss, "b", label = "training loss")
	ax0.set(xlabel='epochs', ylabel='Training loss')
	ax1.plot(epochs, test_loss, "b", label = "test loss")
	ax1.set(xlabel='epochs', ylabel='Test loss')
	ax2.plot(epochs, accuracy, 'r--', label="accuracy")
	ax2.set(xlabel='epochs', ylabel='Accuracy')

	fig.savefig(result_path + "/performamce.png", dpi = 800)


def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)
        batch_preds, loss = model(**data)
        fin_loss += loss.item()
        fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(data_loader)
