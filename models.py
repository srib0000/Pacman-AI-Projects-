import nn
import math

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # List of parameter specifications: (attribute name, dimensions)
        parameters = [
            ('P1', (1, 20)),
            ('Q1', (1, 20)),
            ('P2', (20, 1)),
            ('Q2', (1, 1))
        ]

        # Initializing each parameter dynamically
        for name, dims in parameters:
            setattr(self, name, nn.Parameter(*dims))


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # Layer 1: Linear + ReLU
        x = nn.Linear(x, self.P1)
        x = nn.AddBias(x, self.Q1)
        x = nn.ReLU(x)
        
        # Layer 2: Linear
        x = nn.Linear(x, self.P2)
        x = nn.AddBias(x, self.Q2)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predictions = self.run(x)
        return nn.SquareLoss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        loss = math.inf
        threshold = 0.02
        while loss > threshold:

            x = nn.Constant(dataset.x)
            y = nn.Constant(dataset.y)
            loss_node = self.get_loss(x, y)
            params = [self.P1, self.Q1, self.P2, self.Q2]
            gradients = nn.gradients(loss_node, params)

            # Updating each parameter
            self.P1.update(gradients[0], -0.02)
            self.Q1.update(gradients[1], -0.02)
            self.P2.update(gradients[2], -0.02)
            self.Q2.update(gradients[3], -0.02)

            # Recalculating the loss to check for convergence
            loss = nn.as_scalar(loss_node)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):        
        # List of parameter specifications: (attribute name, dimensions)
        parameters = [
        ('P1', (784, 240)),
        ('Q1', (1, 240)),
        ('P2', (240, 140)),
        ('Q2', (1, 140)),
        ('P3', (140, 10)),
        ('Q3', (1, 10))
                     ]
        # Initializing each parameter dynamically
        for name, dims in parameters:
         setattr(self, name, nn.Parameter(*dims))


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # Layer 1: Linear + ReLU
        x = nn.Linear(x, self.P1)
        x = nn.AddBias(x, self.Q1)
        x = nn.ReLU(x)

        # Layer 2: Linear + ReLU
        x = nn.Linear(x, self.P2)
        x = nn.AddBias(x, self.Q2)
        x = nn.ReLU(x)

        # Layer 3: Linear
        x = nn.Linear(x, self.P3)
        x = nn.AddBias(x, self.Q3)

        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        logits = self.run(x)
        loss = nn.SoftmaxLoss(logits, y)
        return loss


    def train(self, dataset):
        """
        Trains the model.
        """
        # Initializing a variable to store total accuracy
        tot = -math.inf
        # Setting a threshold for accuracy
        threshold = 0.970
        # Continuing training until accuracy surpasses threshold
        while tot < threshold:
            # Iterating through the dataset in batches 
            for x,y in dataset.iterate_once(50): 
                # Computing gradients of loss with respect to model parameters
                gradient_P1, gradient_Q1, gradient_P2, gradient_Q2, gradient_P3, gradient_Q3  = nn.gradients(self.get_loss(x,y), [self.P1, self.Q1, self.P2, self.Q2, self.P3, self.Q3])
                # Updating parameters using gradients with a learning rate of -0.30
                self.P1.update(gradient_P1, -0.30)
                self.Q1.update(gradient_Q1, -0.30)
                self.P2.update(gradient_P2, -0.30)
                self.Q2.update(gradient_Q2, -0.30)
                self.P3.update(gradient_P3, -0.30)
                self.Q3.update(gradient_Q3, -0.30)
                # Getting validation accuracy after each epoch of training
            tot = dataset.get_validation_accuracy()
            

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
