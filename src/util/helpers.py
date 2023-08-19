import torch
from sklearn import preprocessing
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

def CreateTensor(features, labels, device):
    """
    Converts the given features and labels into tensors.

    Args:
        features (numpy.ndarray): Scaled feature matrix.
        labels (pandas.DataFrame): Labels.
        device (str): Device to which tensors will be moved (e.g., 'cpu' or 'cuda').

    Returns:
        features_tensor (torch.Tensor): Tensor containing scaled features.
        labels_tensor (torch.Tensor): Tensor containing labels.
    """
    # Convert features and labels to tensors with the specified device
    features_tensor = torch.tensor(features, dtype=torch.float32, requires_grad=True).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32, requires_grad=True).to(device)
    
    return features_tensor, labels_tensor

def PreprocessData(training_sets):
    """
    Preprocesses the input training data using standard scaling.

    Args:
        training_sets (pandas.DataFrame): The input training data containing columns 't', 'k', 'c', 'm', and 'x'.

    Returns:
        features (numpy.ndarray): Scaled feature matrix containing columns 't', 'k', 'c', and 'm'.
        labels (pandas.DataFrame): Labels from the 'x' column.
    """
    # Initialize a StandardScaler and fit it to the training data
    scaler = preprocessing.StandardScaler().fit(training_sets[['t', 'k', 'c', 'm']])
    features = scaler.transform(training_sets[['t', 'k', 'c', 'm']])
    labels = training_sets[['x']]

    return features, labels

class NeuralNetworkTrainer:
    def __init__(self, model, learning_rate, loss_fn, weight_decay=0.0, patience=5):
        """
        Initialize the NeuralNetworkTrainer class.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            learning_rate (float): Learning rate for the optimizer.
            loss_fn: The custom loss function to be used during training.
            weight_decay (float, optional): Weight decay for the optimizer. Default is 0.0.
            patience (int, optional): Number of epochs for early stopping patience. Default is 5.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.patience = patience

    def train(self, train_loader, num_epochs, t_train, x_train, t_boundary, t_physics, m, c, k):
        """
        Train the neural network model.

        Args:
            train_loader: DataLoader for training data.
            num_epochs (int): Number of training epochs.
            t_train, x_train, t_boundary, t_physics, m, c, k: Your input data.
            
        Returns:
            loss_curve (dict): Dictionary containing training loss curve data.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_loss = float("inf")
        counter = 0

        loss_curve = {'epoch': [], 'loss': []}

        pbar = tqdm(total=num_epochs, desc="Training Progress")
        for epoch in range(num_epochs):
            for features, labels in train_loader:
                optimizer.zero_grad()

                output = self.model(features)
                loss = self.loss_fn(self.model, self.loss_fn.criterion, t_train, x_train, t_boundary, t_physics, m, c, k)

                loss.backward()
                optimizer.step()

                loss_curve['epoch'].append(epoch)
                loss_curve['loss'].append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1

            if counter >= self.patience:
                print(f'Early stopping at epoch {epoch+1} due to lack of loss improvement.')
                break

            pbar.set_postfix({"Loss": loss.item()})
            pbar.update(1)

        return loss_curve

# class CustomLoss:
#     def __init__(self, lam1=1e-1, lam2=1e-4, supervised=True):
#         """
#         Initialize the CustomLoss class.

#         Args:
#             lam1 (float, optional): Weight for loss3. Default is 1e-1.
#             lam2 (float, optional): Weight for loss4. Default is 1e-4.
#             supervised (bool, optional): Whether to use supervised loss. Default is True.
#         """
#         self.lam1 = lam1
#         self.lam2 = lam2
#         self.supervised = supervised

#     def __call__(self, model, criterion, t_train, x_train, t_boundary, t_physics, m, c, k):
#         """
#         Calculate the custom loss.

#         Args:
#             model (torch.nn.Module): The neural network model.
#             criterion: The criterion used for loss calculations.
#             t_train, x_train, t_boundary, t_physics, m, c, k: Your input data.

#         Returns:
#             final_loss: The calculated custom loss.
#         """
#         # General loss
#         if self.supervised:
#             x_pred = model(t_train)
#             loss1 = criterion(x_pred, x_train)
#         else:
#             loss1 = 0

#         # Boundary loss
#         x_pred_boundary = model(t_boundary)
#         loss2 = criterion(x_pred_boundary, torch.ones_like(x_pred_boundary))
#         dxdt = torch.autograd.grad(x_pred_boundary, t_boundary, torch.ones_like(x_pred_boundary), create_graph=True)[0]
#         loss3 = criterion(dxdt, torch.zeros_like(dxdt))

#         # Physics loss
#         x_pred_physics = model(t_physics)
#         dxdt = torch.autograd.grad(x_pred_physics, t_physics, torch.ones_like(x_pred_physics), create_graph=True)[0]
#         d2xdt2 = torch.autograd.grad(dxdt, t_physics, torch.ones_like(dxdt), create_graph=True)[0]
#         h = (m * d2xdt2 + c * dxdt + k * x_pred_physics)
#         loss4 = criterion(h, torch.zeros_like(h))

#         final_loss = loss1 + loss2 + self.lam1 * loss3 + self.lam2 * loss4

#         return final_loss

class NeuralNetworkTrainer:
    def __init__(self, model, learning_rate, loss_fn, weight_decay=0.0, epoch_patience=100, lr_patience=100):
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.epoch_patience = epoch_patience
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=lr_patience, verbose=True)
        self.scaler = GradScaler()

    def train(self, train_loader, num_epochs, features, labels):
        best_loss = float("inf")
        counter = 0

        loss_curve = {'epoch': [], 'loss': []}

        pbar = tqdm(total=num_epochs, desc="Training Progress")
        for epoch in range(num_epochs):
            for features, labels in train_loader:
                self.optimizer.zero_grad()

                loss = self.loss_fn(self.model, features, labels)

                # Backpropagate
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update learning rate
                self.lr_scheduler.step(loss)

                loss_curve['epoch'].append(epoch)
                loss_curve['loss'].append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1

            if counter >= self.epoch_patience:
                print(f'Early stopping at epoch {epoch+1} due to lack of loss improvement.')
                break

            pbar.set_postfix({"Loss": loss.item()})
            pbar.update(1)

        return loss_curve

class CustomLoss:
    def __init__(self, lam1=1e-1, lam2=1e-4, supervised=True):
        self.lam1 = lam1
        self.lam2 = lam2
        self.supervised = supervised

    def __call__(self, model, features, labels):
        criterion = torch.nn.MSELoss()
        # General loss
        if self.supervised:
            output = model(features)
            loss1 = criterion(output, labels)
        else:
            loss1 = 0

        # # Boundary loss (apply only at t = 0)
        # t_zero_indices = (features[:, 0] == 0)
        # loss2 = criterion(output[t_zero_indices], torch.ones_like(output[t_zero_indices]))

        # dxdt = torch.autograd.grad(output[t_zero_indices], features[:, 0], torch.ones_like(output[t_zero_indices]), create_graph=True)[0]
        # loss3 = criterion(dxdt, torch.zeros_like(dxdt)) 

        # # Physics loss
        # output = model(features)
        # dxdt = torch.autograd.grad(output, features['t'], torch.ones_like(output), create_graph=True)[0]
        # d2xdt2 = torch.autograd.grad(dxdt, features['t'], torch.ones_like(dxdt), create_graph=True)[0]
        # h = (features['m'] * d2xdt2 + features['c'] * dxdt + features['k'] * output)
        # loss4 = torch.mean(h**2)

        # final_loss = loss1 + loss2 + self.lam1 * loss3 + self.lam2 * loss4
        # final_loss = loss1 + loss2 + self.lam1 * loss3
        final_loss = loss1

        return final_loss
    
