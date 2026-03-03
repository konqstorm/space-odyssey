Reinforce
First iteration: where bad model inputs, just raw simulation data - coordinates, velocities.  Resulted in a spinner from hell which was getting giant torque and then moving randomly

Second iteration: updated inputs to make them alligned to the ship and show error from target, changed the model to have separate fc layers before the heads and also separated torque and thrust control to different heads. Now resulted in more adequate, but still somewhat random left-right swinging pattern

Third iteration: switched the reward function for distance to be non-lineary dependant on the distance to the target, more potential-like (closer to target - bigger reward for improvement). Result: now sometimes when he is close to the target he tries to carefully adjust the position, but when far away he switches back to idiot swing mode