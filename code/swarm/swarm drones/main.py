if __name__ == '__main__':
    pid_class = PID()
    error = []
    # Go to intial set point
    while True:
        error = pid_class.pid(pid_class.initialSetPoint)
        if (abs(error[0]) <= 0.5 and abs(error[1]) <= 0.5 and abs(error[2]) <= 0.5):
            break
    #go to one by one checkpoint
    for i in range (0, 3):
        if i != 0:
            print("next_target")
        pid_class.next.publish('1') #signal to sent next check point
        rospy.sleep(0.5)
        length = len(pid_class.setPoint.poses)
        for j in range (0, length): #go to one by one subsetpoint
            temp = [0.0, 0.0, 0.0, 0.0]
            temp[0] = pid_class.setPoint.poses[j].position.x
            temp[1] = pid_class.setPoint.poses[j].position.y
            temp[2] = pid_class.setPoint.poses[j].position.z
            temp[3] = 0.0
            while (True): # while constrained not meet
                error = pid_class.pid(temp)
                if (error[0] <= 0.5 and error[1] <= 0.5 and error[2] <= 0.5):
                    break
    #land at start point
    while True:
        error = pid_class.pid(pid_class.start) #find error start point
        # if constrained meet then disarm (i.e disarm near ground)
        if (abs(error[0]) <= 0.5 and abs(error[1]) <= 0.5 and abs(error[2]) <= 1):
            pid_class.disarm()
            break