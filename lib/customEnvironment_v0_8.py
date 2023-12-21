#################################################
# Imports
#################################################

import os
import time
import airsim
import numpy as np
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


#################################################
# AirSim environment definition
#################################################

class DroneEnvironment(py_environment.PyEnvironment):

  '''Drone'a bağlanarak ve gözlem ve eylem alanini ayarlayarak ortami başlatir
  :param enable_wind: rüzgar etkinliği
  :param randomize_initial_pose: drone'nun ilk konumunun ve yönünün rastgele ayarlanmasi
  :param save_path: veri kaydetme klasörünün yolu:
                      Varsa, yazilim daha sonra analiz için dronun tüm durumlarini kaydeder
                      Yok ise hiçbir şey kaydedilmez
  '''
  def __init__(self, enable_wind=False, randomize_initial_pose=False, save_path=None, decrease_unreal_load=False):
    self.enable_wind = enable_wind
    self.randomize_initial_pose = randomize_initial_pose
    self.save_path = save_path
    self._states_arr = None

    self.client = airsim.MultirotorClient()
    if decrease_unreal_load: self.client.simRunConsoleCommand("t.MaxFPS 5")
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    
    self._observation_spec = array_spec.ArraySpec(shape=(19,), dtype=np.float32, name='observation') # gözlem kısmı
    self._action_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, name='action', minimum=0.0, maximum=1.0) # eylem kısmı: motor güç kontrolleri
    
    #self._state, _, _, _, _, _, _ = self.getState()
    self._episode_ended = False
    self._total_reward = 0
  
  def action_spec(self):
    return self._action_spec
  
  def observation_spec(self):
    return self._observation_spec



  '''Rastgele veya rastgele olmayan yeni bir poz oluşturur
  '''
  def getNewPose(self, random=False, random_uniform=False):
    pos_stddev = 0.25 # in [m]
    or_stddev = 0.3 # in [rad], 0.15 -> en fazla 20 derecelik eğim
    if random:
      if random_uniform:
        u = np.random.uniform()
        v = np.random.uniform()
        w = np.random.uniform()
        new_pose = airsim.Pose(position_val=airsim.Vector3r(0.0 + np.random.normal(0, pos_stddev), 0.0 + np.random.normal(0, pos_stddev), -100.0 + np.random.normal(0, pos_stddev)),
                                orientation_val=airsim.Quaternionr(np.sqrt(1-u)*np.sin(2*np.pi*v), np.sqrt(1-u)*np.cos(2*np.pi*v), np.sqrt(u)*np.sin(2*np.pi*w), np.sqrt(u)*np.cos(2*np.pi*w)))
      else:
        new_pose = airsim.Pose(position_val=airsim.Vector3r(0.0 + np.random.normal(0, pos_stddev), 0.0 + np.random.normal(0, pos_stddev), -100.0 + np.random.normal(0, pos_stddev)),
                                #orientation_val=airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)) # do not put noise in orientation
                                orientation_val=airsim.utils.to_quaternion(np.random.normal(0, or_stddev), np.random.normal(0, or_stddev), np.random.normal(0, or_stddev))) # radyan cinsinden yuvarlanma eğimi
    else:
      new_pose = airsim.Pose(position_val=airsim.Vector3r(0.0, 0.0, -100.0), orientation_val=airsim.Quaternionr(0.0, 0.0, 0.0, 1.0))
    reference_pose = airsim.Pose(position_val=airsim.Vector3r(0.0, 0.0, -100.0), orientation_val=airsim.Quaternionr(0.0, 0.0, 0.0, 1.0))
    return new_pose, reference_pose
  
  '''Drone'nun pozunu işlevin içinde belirtilene sifirlar ve multirotorun durumunu yazdirir (uçuyor ya da uçmuyor)
  '''
  def reset_pose(self):
    self.client.reset()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    self.client.takeoffAsync(timeout_sec=0.1, vehicle_name="SimpleFlight").join() # çok rotorlu doğru uçuş durumunu etkinleştirmek için gerekli

    new_pose, self.initial_pose = self.getNewPose(self.randomize_initial_pose)

    self.client.simSetVehiclePose(pose=new_pose, ignore_collision=False, vehicle_name="SimpleFlight")
    if self.client.getMultirotorState().landed_state == airsim.LandedState.Landed: print("[LANDED: Physics Engine NOT Engaged]")
    else: print("[CORRECTLY FLYING: Physics Engine Engaged]")
    time.sleep(0.01) # gerekli! çünkü sinir ağı çok hızlı ve adımlar çok erken başlar,ve drone hazır değilken hiçbir deneyim kazanılamaz (rew=0)
  
  '''Simülasyonda verilen standart sapma ile rastgele bir rüzgar ayarlar. [m]
  '''
  def setRandomWind(self, stddev=2.5):
    x_val = np.random.normal(0, stddev)
    y_val = np.random.normal(0, stddev)
    z_val = np.random.normal(0, stddev)
    wind = airsim.Vector3r(x_val, y_val, z_val)
    print('Wind set < x y z >: <', x_val, y_val, z_val, '>')
    self.client.simSetWind(wind)
  
  '''Oluşturulan özel ortami sifirlar
  '''
  def _reset(self):
    print('Total reward for the previous episode:', self._total_reward)
    self._total_reward = 0
    self._steps = 0
    if self.save_path is not None: # analiz için dronun durumlarını kaydeder
      if self._states_arr is not None:
        if not os.path.exists(self.save_path+'/states'): os.makedirs(self.save_path+'/states')
        np.save(self.save_path+'/states/'+str(time.time()), self._states_arr)
      self._states_arr = np.empty((0,19))

    if self.enable_wind: self.setRandomWind()
    self.reset_pose()

    self._state, _, _, _, _, _, _ = self.getState()
    self._episode_ended = False
    return ts.restart(self._state)



  '''Önce sonlandirma koşullarini kontrol ederek drone'u eylem tarafindan belirtildiği şekilde hareket ettirir
  :param action: tensorflow eylemi, __init__ function dizi özelliği tarafindan belirtildiği gibi
  :param duration: eylemin süresi ne kadar olmalidir. Hareket sürekli ise, yalnizca maksimum süreyi belirtir (zaman uyumsuzdur)
  :param continuous:  ağ çikarimina geri dönmeden önce sürekli hareketler yapmak veya eylemin bitmesini beklemek
  :return: Bölüm, çarpişmalar veya diğer nedenlerle sona ermek zorundaysa True, aksi takdirde False
  '''
  def move(self, action, duration=1.002, continuous=True):
    #if self.client.simGetCollisionInfo().has_collided or self.client.simGetVehiclePose().position.z_val > -10: return True

    # birleştirerek rotor itme kuvveti sonraki süre için varsayılan sabit değer 2.42 ye sıfırlanır. Olmazsa ağın karar verdiği şeye bırakılır
    if continuous == True: # sürekli hareketler -> kontrol döngüsü : ağ çıkarımı -> eylemi eşzamansız olarak gerçekleştir -> ...
      scale = 2.5 # her motor için delta değerleri ölçeği, ne kadar değer verileceği
      b_th = 0.59
      d_th = scale * (action-0.5) / 5 # rotorlarda delta itiş
      th = np.clip(b_th + d_th, 0, 1) # itme sapması + delta, kırpılmış
      self.client.moveByMotorPWMsAsync(front_right_pwm=float(th[0]), rear_left_pwm=float(th[1]), front_left_pwm=float(th[2]), rear_right_pwm=float(th[3]), duration=duration)
      #th=action
      #self.client.moveByMotorPWMsAsync(front_right_pwm=0.0, rear_left_pwm=float(th[1]), front_left_pwm=float(th[2]), rear_right_pwm=float(th[3]), duration=duration)
      time.sleep(0.002)

    else: # ayrık hareketler -> kontrol döngüsü : ağ çıkarımı -> eylemi gerçekleştir ve katıl -> ...
      #self.client.hoverAsync()
      scale = 2.0 # her motor için delta değerleri ölçeği, ne kadar değer verileceği
      b_th = 0.59
      d_th = scale * (action-0.5) / 5 # rotorlarda delta itiş
      th = np.clip(b_th + d_th, 0, 1) # itme sapması + delta, kırpılmış
      self.client.moveByMotorPWMsAsync(front_right_pwm=float(th[0]), rear_left_pwm=float(th[1]), front_left_pwm=float(th[2]), rear_right_pwm=float(th[3]), duration=duration).join()
    
    return False

  '''Geçersiz kil, bir adim: bir eylem gerçekleştirir, ödülü alir, ilgili geçiş veya sonlandirma sinyallerini döndürür
  :param action: araci tarafindan kararlaştirilan gerçekleştirilecek eylem
  :return: ya bir geçiş ya da bir sonlandirma
  '''
  def _step(self, action):
    # Bölüm bittiyse, ortamı sıfırlar
    if self._episode_ended: return self.reset()

    # seçilen hareketi gerçekleştirir, çarpışma olursa True döndürür, yeni durum elde eder, ödülü hesaplar
    end_now = self.move(action=action)
    self._state, pos, orient, ang_acc, ang_vel, lin_acc, lin_vel = self.getState()
    if self.save_path is not None: self._states_arr = np.concatenate((self._states_arr, [self._state]), axis=0) # save the states of the drone for later analysis
    reward = self.reward_function(pos, orient, ang_acc, ang_vel, lin_acc, lin_vel)

    # İşlem durumları - step_type: 0->beginning, 1->normal transition, 2->terminal
    if end_now:
      print('Collision occurred or episode termination condition met')
      self._episode_ended = True
      reward = 0 #  drone çarpınca
      return ts.termination(self._state, reward=reward) # terminale ajanın durumunu döndürür
    else: # halen devam ediyorsa girer
      self._total_reward += reward
      return ts.transition(self._state, reward=reward)



  '''Durumu, float32 değerlerine sahip bir numpy dizi olarak döndürür
  '''
  def getState(self):
    state   = self.client.getMultirotorState()
    pos     = state.kinematics_estimated.position
    orient  = state.kinematics_estimated.orientation
    ang_acc = state.kinematics_estimated.angular_acceleration
    ang_vel = state.kinematics_estimated.angular_velocity
    lin_acc = state.kinematics_estimated.linear_acceleration
    lin_vel = state.kinematics_estimated.linear_velocity
    
    return np.array([(pos.z_val-self.initial_pose.position.z_val)/10, (pos.x_val-self.initial_pose.position.x_val)/10, (pos.y_val-self.initial_pose.position.y_val)/10, # between -1 and 1 more or less
                    orient.w_val, orient.x_val, orient.y_val, orient.z_val, # between -1 and 1
                    ang_acc.x_val/10, ang_acc.y_val/10, ang_acc.z_val/10,   # at most around 30ish?
                    ang_vel.x_val/10, ang_vel.y_val/10, ang_vel.z_val/10,   # at most around 30ish?
                    lin_acc.x_val/10, lin_acc.y_val/10, lin_acc.z_val/10,   # at most around 30ish?
                    lin_vel.x_val/10, lin_vel.y_val/10, lin_vel.z_val/10],  # at most around 30ish?
                    dtype=np.float32), pos, orient, ang_acc, ang_vel, lin_acc, lin_vel
  
  '''Poz verildiğinde ödülü verir (durum)
  '''
  def reward_function(self, pos, orient, ang_acc, ang_vel, lin_acc, lin_vel):

    reward = max(0, 1 - np.sqrt((pos.z_val-self.initial_pose.position.z_val)**2 + (pos.x_val-self.initial_pose.position.x_val)**2 + (pos.y_val-self.initial_pose.position.y_val)**2))
    reward -= 0.1 * np.sqrt((orient.w_val-1)**2 + orient.x_val**2 + orient.y_val**2 + orient.z_val**2)
    reward -= 0.1 * np.sqrt(ang_vel.x_val**2 + ang_vel.y_val**2 + ang_vel.z_val**2)

    self._steps += 1
    #if self._steps % 10 == 0: print('Position of drone: <', pos.x_val, pos.y_val, pos.z_val, '>')

    return reward