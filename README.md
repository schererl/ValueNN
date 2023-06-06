# ValueNN
Exercise for understanding how ValueNetworks models are trained using a simplified version of Amazons

## Referencial Teórico
Este projeto tem como intuito utilizar de Redes Neurais para gerar um modelo capaz de jogar Amazons. As principais tecnologias utilizadas foram python e a biblioteca pytorch.

### Amazons
É um jogo de tabuleiro composto por, um tabuleiro de 10x10 posições e dois jogadores com quatro peças cada. Estas peças tem o movimento da Dama do jogo de damas (ou a combinação entre Torre e Bispo do xadrez), ou seja move-se quantas posições desejar nas oito direções possiveis, alem de poder lançar uma lança, em qualquer direção seguindo a mesma logica de seu movimento, estas lanças nao são removidas durante o jogo. O objetivo do jogo é imobilizar todas as peças do adversario.

### Definição do Estado de Jogo
Para facilitar e agilizar no processo de treinamento e aprendizado o espaço de estados do jogo foi reduzido pela metade, agora sendo representado por um tabuleiro de 5x5 posições e 2 jogadores. Um estado de jogo é a representação de uma das possiveis combinações de jogo no tabuleiro e as peças posicionadas sobre ele, para a rede neural o que envia-se é um vetor, onde a primeira posição representa o jogador atual e as demais representam o tabuleiro e suas peças.

## Delimitação do Desenvolvimento

## Resultados e Conclusão

## Referências
