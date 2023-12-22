#################################################
# Imports
#################################################
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


#################################################
# Reinforcement Learning parameters
#################################################
sequence = 0
save_path = "c:/Users/ELMALI/Desktop/proje_v5/src/training_data/20230120_214304-TD3-1000/states"
directory = "c:/Users/ELMALI/Desktop/proje_v5/src/training_data/20230120_214304-TD3-1000/plots"
if not os.path.exists(directory):
            os.makedirs(directory)
for dump in os.listdir(save_path):
    sequence = sequence + 1 
    print(sequence)
    states = np.load(save_path + '/' + dump)
    # delta_z/10, delta_x/10, delta_y/10, orient.w_val, orient.x_val, orient.y_val, orient.z_val, ang_acc.x_val/10, ang_acc.y_val/10, ang_acc.z_val/10,
    # ang_vel.x_val/10, ang_vel.y_val/10, ang_vel.z_val/10, lin_acc.x_val/10, lin_acc.y_val/10, lin_acc.z_val/10, lin_vel.x_val/10, lin_vel.y_val/10, lin_vel.z_val/10
    print(states.shape)

    
    rotations = []
    for i in range(len(states)):
        rot = R.from_quat([states[i,4], states[i,5], states[i,6], states[i,3]])
        rotations.append(rot.as_euler('xyz', degrees=True))
    rotations = np.array(rotations)
    
    figure = plt.figure(figsize=(18, 18)) 
    figure.tight_layout(pad=5.0)            
    figure.suptitle('Sequence: ' + str(sequence))


    plt.subplot(221)
    plt.title("roll pitch yaw degrees")
    plt.grid()
    plt.plot(rotations[:,0], label="roll")
    plt.plot(rotations[:,1], label="pitch")
    plt.plot(rotations[:,2], label="yaw")
    #plt.savefig(directory + '/' + str(sequence) + ' roll pitch yaw degrees')
    plt.legend()
    
    plt.subplot(222)
    plt.title("lin velocity integrated m/s")
    plt.grid()
    plt.plot(10*states[:,16],label="x lin vel")
    plt.plot(10*states[:,17],label="y lin vel")
    plt.plot(10*states[:,18],label="z lin vel")
    #plt.savefig(directory + '/' + str(sequence) + ' lin velocity integrated m/s')
    plt.legend()
    
    plt.subplot(223)
    plt.title("lin acc m/s2?")
    plt.grid()
    plt.plot(10*states[:,13],label="x acc")
    plt.plot(10*states[:,14],label="y acc")
    plt.plot(10*states[:,15],label="z acc")
    #plt.savefig(directory)
    plt.legend()
    
    plt.subplot(224)
    plt.title("angular acc")
    plt.grid()
    plt.plot(10*states[:,7],label="x ang acc")
    plt.plot(10*states[:,8],label="y ang acc")
    plt.plot(10*states[:,9],label="z ang acc")
    #plt.savefig(directory + '/' + str(sequence) + " angular acc")
    plt.legend()

    plt.savefig(directory + '/' + str(sequence))
    
    
    
    

'''
dump_path =os.getcwd() + "C:/Users/ELMALI/Desktop/proje_v5/src/training_data/20230120_130531-SAC-1000/states/1674209300.4010832.npy"
states_drone = np.load(dump_path)
# x_roll y_pitch z_yaw v_x v_y v_z acc_x acc_y acc_z gyr_x gyr_y gyr_z gyr_x gyr_y acc_x acc_y acc_z gyr_x gyr_y gyr_z gyr_offset [vision vel x y z]
print(states_drone.shape)


plt.title("pitch roll yaw degrees")
plt.plot(states_drone[:,0], label="roll")
plt.plot(states_drone[:,1], label="pitch")
plt.plot(states_drone[:,2], label="yaw")
plt.legend()
plt.show()

plt.title("lin velocity mm/s")
plt.plot(states_drone[:,3], label="x")
plt.plot(states_drone[:,4], label="y")
plt.plot(states_drone[:,5], label="z")

plt.title("lin velocity vision")
plt.plot(states_drone[:,21], label="vis_x")
plt.plot(states_drone[:,22], label="vis_y")
plt.plot(states_drone[:,23], label="vis_z")
plt.legend()
plt.show()

plt.title("acc lsb")
plt.plot(states_drone[:,6], label="acc_x")
plt.plot(states_drone[:,7], label="acc_y")
plt.plot(states_drone[:,8], label="acc_z")
plt.legend()
plt.show()

def smooth(x):
    x = np.pad(x, (0, 2), 'constant', constant_values=(0, 0))
    x = np.reshape(x, (-1, 4))
    x = np.average(x, axis=1)
    return x
def filter(x):
    return x[abs(x) < 1e2]

print(len(states_drone[:,14]))
print(len(filter(states_drone[:,14])))

plt.title("acc phys")
#plt.hist(states_drone[:,14], bins=1000)
plt.plot(filter(states_drone[:,14]), label="acc_x")
plt.plot(filter(states_drone[:,15]), label="acc_y")
plt.plot(filter(states_drone[:,16]), label="acc_z")
plt.legend()
plt.show()

plt.title("gyro phys")
#plt.hist(states_drone[:,14], bins=1000)
plt.plot(filter(states_drone[:,17]), label="gyro_x")
plt.plot(filter(states_drone[:,18]), label="gyro_y")
plt.plot(filter(states_drone[:,19]), label="gyro_z")
plt.legend()
plt.show()
'''
