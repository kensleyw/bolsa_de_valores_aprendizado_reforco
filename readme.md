# Projeto de Aprendizado por Reforço para Mercado Financeiro

Este projeto utiliza técnicas de Aprendizado por Reforço para treinar um agente a tomar decisões de compra e venda em um ambiente simulado de mercado financeiro. O agente é treinado em um ambiente personalizado chamado **MercadoFinanceiro**, onde as ações disponíveis são comprar (0) ou vender (1). O objetivo é maximizar o lucro ao longo do tempo.

## Ambiente de Mercado Financeiro
O ambiente MercadoFinanceiro é uma classe que herda da interface **gym.Env** do OpenAI Gym. Ela representa um ambiente de mercado financeiro básico, onde um agente pode realizar ações de compra e venda com base em uma série temporal de preços. O ambiente inclui métodos como **reset** para reiniciar o episódio, **step** para executar uma ação e avançar no tempo, e **_get_observation** para obter a observação atual.


Projeto de Aprendizado por Reforço para Mercado Financeiro
Este projeto utiliza técnicas de Aprendizado por Reforço para treinar um agente a tomar decisões de compra e venda em um ambiente simulado de mercado financeiro. O agente é treinado em um ambiente personalizado chamado MercadoFinanceiro, onde as ações disponíveis são comprar (0) ou vender (1). O objetivo é maximizar o lucro ao longo do tempo.

Ambiente de Mercado Financeiro
O ambiente MercadoFinanceiro é uma classe que herda da interface gym.Env do OpenAI Gym. Ela representa um ambiente de mercado financeiro básico, onde um agente pode realizar ações de compra e venda com base em uma série temporal de preços. O ambiente inclui métodos como reset para reiniciar o episódio, step para executar uma ação e avançar no tempo, e _get_observation para obter a observação atual.

## Modelo do Agente
O agente utiliza uma política representada por uma rede neural simples implementada com a biblioteca TensorFlow. A política é treinada usando o algoritmo de otimização Proximal Policy Optimization (PPO). A arquitetura da rede neural consiste em duas camadas ocultas de 64 unidades cada, seguidas por uma camada de saída com o número de ações no espaço de ação do ambiente.

## Treinamento do Agente
O treinamento do agente ocorre em um loop principal onde ele interage com o ambiente, coleta observações e ações, calcula gradientes usando a loss definida, e atualiza os parâmetros da política usando o otimizador Adam. O treinamento é repetido por um número específico de iterações (**num_steps**).

## Teste Final do Agente
Após o treinamento, o agente é testado em um ambiente de mercado financeiro semelhante para avaliar seu desempenho. O resultado final é o lucro total obtido durante o episódio de teste e o percentual de lucro em relação ao investimento inicial.

## Executando o Projeto
Antes de executar o projeto, certifique-se de ter as bibliotecas necessárias instaladas, incluindo **gym, numpy, yfinance, time, tensorflow e stable_baselines3**. Você pode instalar essas dependências usando:

```bash
pip install gym numpy yfinance tensorflow stable-baselines3[extra]
```
Certifique-se também de ter as permissões necessárias para acessar o ambiente de mercado financeiro.

```python
# Executando o projeto

# Obtém a série histórica de ações do Banco do Brasil (BBAS3)
data = yf.download('BBAS3.SA', start='2021-12-25', end='2024-01-05')['Close'].values

# Normaliza os preços para o intervalo [0, 1]
normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Criação do ambiente
env = MercadoFinanceiro(normalized_data)

# Definindo o valor do investimento inicial
investimento_inicial = 10000

# Treinamento do agente
policy = train_agent(env, num_steps=1000) # num_steps quantas repetições de teste

# Teste final do agente
final_profit = test_agent(env, policy)
percentual_lucro = (final_profit / investimento_inicial + 1) * 100

print(f"Lucro final em valor: R$ {final_profit:.2f}")
print(f"Lucro final em percentual: {percentual_lucro:.2f}%")

```
Este script demonstra a criação do ambiente, treinamento e teste do agente em um ambiente simulado de mercado financeiro. Experimente ajustar os parâmetros e a arquitetura da rede neural para otimizar o desempenho do agente.

## Colaboração
Contribuições são bem-vindas! Se você tem sugestões de melhorias, encontrou bugs ou deseja adicionar novos recursos, sinta-se à vontade para criar issues ou pull requests. Vamos trabalhar juntos para aprimorar este projeto de aprendizado por reforço no mercado financeiro.
