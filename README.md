# Sequential-Learning
A computational model of Cortico-Striato-Thalamic circuits is used for action selection. The model, originally proposed by Şengör, et al.(2008), is depicted in figures. To investigate the convenience of the model, the task of learning a sequence of stimulus-action pairs defined below is considered. 

![image](https://user-images.githubusercontent.com/75907241/129909795-1169abe5-9e9c-4d13-82f8-b0ac2cb837d3.png)

![image(1)](https://user-images.githubusercontent.com/75907241/129909865-19c1e641-bbcb-47c6-8b39-3497fb806e84.png)

The training phase has three stages. First stimulus C is given and action 3 is desired and a reward is given if the subject does the true action. This is called the C-3 stage. Then B is given and 2 is desired. If the subject does the true action then no reward but C is given and 3 is desired. A reward is given if the subject does the true action. This is called B-2 C-3 stage. Then comes the A-1 B-2 C-3 stage.

Şengör, N. S., Karabacak, O., &  Steinmetz, U. (2008, September). A computational model of  cortico-striato-thalamic circuits in goal-directed behaviour. In International Conference on Artificial Neural Networks (pp. 328-337). Springer, Berlin, Heidelberg.
