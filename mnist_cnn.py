import torch.nn as nn

# Laver funktion der laver et convolutional neural network (cnn)
def cnn():
    cnn = nn.Sequential(
        # 1. lag: Convolutional Layer
            # 1 input-kanal: Dimension af farve- eller intensitetsinformation; 1, da vi arbejder med gråtoner
            # 10 outputkanaler: Antal filtre
            # Kernel_size=3: Størrelsen på det udsnit af billedet vi tager, som således bliver en 3x3 matrice af pixels.
            # Benytter ReLU-funktion: Sætter negative værdier til 0 og beholder positive værdier uændret 
                # Dimension af tensorerne/billederne her bliver 28 - 3 + 1 = 26
        nn.Conv2d(1, 10, kernel_size=3), 
        nn.ReLU(),

        # 2. lag: Max-pooling Layer
            # Kernel_size=2: Vælger den maksimale værdi i et udsnit af størrelsen 2x2 pixels, således dimensionerne af billederne halveres
                # Dimension af tensorerne her bliver 26 / 2 = 13
        nn.MaxPool2d(kernel_size=2),

        # 3. lag: Convolutional Layer
            # 10 input-kanaler: Da 1. lag havde 10 out-put kanaler, og 2. lag ændrede ikke dette
            # 10 outputkanaler: Antal filtre
            # Kernel_size=3: Størrelsen på det udsnit af billedet vi tager, som således bliver en 3x3 matrice af pixels.
            # Benytter ReLU-funktion: Sætter negative værdier til 0 og beholder positive værdier uændret 
                # Dimension af tensorerne/billederne her bliver 13 - 3 + 1 = 11
        nn.Conv2d(10, 10, kernel_size=3),
        nn.ReLU(),

        # 4. lag: Max-pooling Layer
            # Kernel_size=2: Vælger den maksimale værdi i et udsnit af størrelsen 2x2 pixels, således dimensionerne af billederne halveres
                # Dimension af tensorerne/billederne her bliver 11 / 2 = 5 (anvender heltalsdivision)
        nn.MaxPool2d(kernel_size=2), 

        # 5. lag: Flatten
            # Omdanner det multidimensionelle data (3D tensor på formen tensor(batch_size, channels, height, width)) til en 1D vektor
                # Dimension af tensoren her bliver (5 * 5) * 10 = 250
        nn.Flatten(),

        # 6. lag: Fully Connected Layer
            # Tager 1D vektoren fra 5. lag, som er i 250 dimensioner, og producerer 10 output der hver repræsenterer et tal fra 0-9
        nn.Linear(250,10), # input er nu 5 x 5 x 10
    )
    return cnn

def cnn_dropout():
    cnn = nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=3), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25), # Tilføjer dropout-lag med dropout-rate på 25% - 25% af tilfældige neuroner i dette lag sættes til værdien 0
        nn.Conv2d(10, 10, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), 
        nn.Dropout(0.25), # Tilføjer dropout-lag med dropout-rate på 25% - 25% af tilfældige neuroner i dette lag sættes til værdien 0
        nn.Flatten(),
        nn.Linear(250,10), 
    )
    return cnn