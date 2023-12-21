#################################################
# Imports
#################################################
import os
import time
import datetime
import numpy as np
import tensorflow as tf
from lib.plotters import Plotter
from lib.customEnvironment_v0_8 import DroneEnvironment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import TimeLimit
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents import Td3Agent
from tf_agents.policies import policy_saver
from tf_agents.utils import common

np.random.seed(1234)
tf.random.set_seed(12345)


#################################################
# Reinforcement Learning parameters
#################################################

save_path = os.getcwd() + '/training_data/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# veri toplama
replay_buffer_capacity = 1000000
initial_collect_steps = 100 # rastgele bir ilkeyle toplanan toplam adım sayısı. TimeLimit adımlarına her ulaşıldığında, ortam sıfırlanır

# Agent
fc_layer_params = (128, 128,)

# Training
train_env_steps_limit = 200 # eğitim ortamının TimeLimit'indeki maksimum adım sayısı
collect_steps_per_iteration = 200 # her bölümdeki maksimum adım sayısı

epochs = 1000
batch_size = 512
learning_rate = 3e-4
checkpoint_dir = save_path + '/ckpts'
policy_dir = save_path + '/policies'
ckpts_interval = 10 # eğitim sırasında depolamak için kaç epochs da bir kontrol noktası olacağı

# Evaluation
eval_env_steps_limit = 400 # ölçüm ortamının TimeLimit'indeki maksimum adım sayısı
num_eval_episodes = 5
eval_interval = 50 # ölçüm ve policy kaydetme aralığı, = yalnızca sonunda ölçüm için epochs


#################################################
# Environments instantiation
#################################################

tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(False, False), duration=train_env_steps_limit)) # ortamdaki n adıma sınır ayarla
eval_tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(DroneEnvironment(False, False, save_path), duration=eval_env_steps_limit)) # ortamdaki m adımlara sınır ayarla


print(tf.ones(tf_env.time_step_spec().observation.shape))

#################################################
# Agent
#################################################

global_step = tf.compat.v1.train.get_or_create_global_step() # adımların küresel sayacı

actor_net = ActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=fc_layer_params, activation_fn=tf.keras.activations.tanh)
critic_net = CriticNetwork((tf_env.observation_spec(), tf_env.action_spec()), joint_fc_layer_params=fc_layer_params, activation_fn=tf.keras.activations.relu)

# Ajan için çalıştırılacak algoritmanın belirlenmesi

agent = Td3Agent(tf_env.time_step_spec(),
                  tf_env.action_spec(),
                  actor_network=actor_net,
                  critic_network=critic_net,
                  actor_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  critic_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  target_update_tau=1.0,
                  target_update_period=2,
                  gamma=0.99,
                  train_step_counter=global_step)

agent.initialize()

print("\nActor network summary and details")
print(actor_net.summary())
for i, layer in enumerate (actor_net.layers):
    print (i, layer)
    try: print ("    ",layer.activation)
    except AttributeError: print('   no activation attribute')

print("\nCritic network summary and details")
print(critic_net.summary())
for i, layer in enumerate (critic_net.layers):
    print (i, layer)
    try: print ("    ",layer.activation)
    except AttributeError: print('   no activation attribute')


#################################################
# Replay Buffer & Collect Driver
#################################################

# İlk toplama politikası - rastgele
tf_policy = random_tf_policy.RandomTFPolicy(action_spec=tf_env.action_spec(), time_step_spec=tf_env.time_step_spec())

# Tekrar arabelleğini oluştur
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=tf_env.batch_size, max_length=replay_buffer_capacity)

# İlk ve eğitim toplama sürücülerini oluşturun
num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
observers = [replay_buffer.add_batch, num_episodes, env_steps]
collect_driver = dynamic_step_driver.DynamicStepDriver(tf_env, tf_policy, observers=observers, num_steps=initial_collect_steps) # tf_policy kullan, hangisi rastgele

train_driver = dynamic_step_driver.DynamicStepDriver(tf_env, agent.collect_policy, observers=observers, num_steps=collect_steps_per_iteration) # tf_policy yerine OUNoisePolicy olan agent.collect_policy'yi kullanın

# İlk veri toplama
print('\nCollecting initial data')
collect_driver.run()
print('Data collection executed\n')

# Tekrar Oynatma Arabelleğini Veri Kümesine Dönüştür
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3) # her biri 2 zaman adımına sahip 512 öğeden oluşan yığınları okuyun
iterator = iter(dataset)


#################################################
# Training and Evaluation functions
#################################################

train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir, max_to_keep=1, agent=agent, policy=agent.policy, replay_buffer=replay_buffer, global_step=global_step)
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

data_plotter = Plotter()

def train_one_iteration():
  start = time.time()
  train_driver.run() # Collect_policy'yi kullanarak birkaç adım toplayın ve yeniden oynatma arabelleğine kaydedin
  end = time.time()
  experience, unused_info = next(iterator) # arabellekten bir grup veriyi örnekleyin ve aracının ağını güncelleyin
  with tf.device('/CPU:0'): train_loss = agent.train(experience) # trains de 1 batch deneyimle
  iteration = agent.train_step_counter.numpy()
  #data_plotter.update_loss(train_loss.loss)
  print ('Iteration:', iteration)
  print('Total_loss:', float(train_loss.loss), 'actor_loss:', float(train_loss.extra.actor_loss), 'critic_loss:', float(train_loss.extra.critic_loss))
  print('Control loop timing for 1 timestep [s]:', (end-start)/collect_steps_per_iteration)

def evaluate_agent(policy, eval_tf_env, num_eval_episodes):
  print('\nEVALUATING *******\n')
  total_reward = 0
  for idx in range(num_eval_episodes):
    print('Evaluation iteration:', idx)
    start = time.time()
    time_step = eval_tf_env.reset()
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = eval_tf_env.step(action_step.action)
      total_reward += float(time_step.reward)
    end = time.time()
    print('Control loop timing for 1 timestep [s]:', (end-start)/eval_env_steps_limit)
  print('\n******* EVALUATION ENDED\n')
  return total_reward / num_eval_episodes # her bölümde ödül avarajı alır

# Eğitim döngüsü, ölçüm & kontrol noktalarının kaydı
avg_rewards = np.empty((0,2))
for epoch in range(epochs+1):
  train_one_iteration()
  if epoch % ckpts_interval == 0:
    train_checkpointer.save(global_step)
  if epoch % eval_interval == 0:
    tf_policy_saver.save(policy_dir+'/'+str(time.time())) # politika restore edildikten sonra kaydedilir
    avg_rew = evaluate_agent(agent.policy, eval_tf_env, num_eval_episodes)
    avg_rewards = np.concatenate((avg_rewards, [[epoch, avg_rew]]), axis=0)
    data_plotter.update_eval_reward(avg_rew, eval_interval)

np.save(save_path+'/avg_rewards.npy', avg_rewards)

data_plotter.plot_evaluation_rewards(avg_rewards, save_path)

# Restoring a checkpoint
#train_checkpointer.initialize_or_restore()
#global_step = tf.compat.v1.train.get_global_step()

# Restoring only the policy
#saved_policy = tf.saved_model.load(policy_dir)