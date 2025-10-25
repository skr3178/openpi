## Intro FLow matching
Flow matching is added to the end of the VLAction models- version2 (as per talk by Sergey Levine). 

Predict the flow path that takes it from the initial to the final position. 
Classically used in image gen, language token gen, etc
Why use for robotics? 

Smoothened, better control- continous outputs for high frequency appplications. 
Advantages over regular diffusion is that it can be completed in less number of steps. 

## Training process

1. sample epsilon from a normal gaussian as random noise 
2. Noisy actions A_t_tau= tau* A_t + (1- tau)* epsilon
3. Train to minimize the conditional flow matching loss 
Network output: v_theta(At_tau, o_t) <==> u(A_t_tau|A_t), also given as epsilon - A_t denoising vector field output


## Here are some of the REFERENCES: 

1. https://diffusionflow.github.io/
core explanation from deepmind

2. https://federicosarrocco.com/blog/flow-matching
Comprehensive codebase 

3. https://www.laumy.tech/2680.html/flow-matching%EF%BC%9A%E8%AE%A9%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E6%B5%81%E5%8A%A8%E8%B5%B7%E6%9D%A5/
Simplified code

4. https://jmtomczak.github.io/blog/18/18_fm.html
Math infused explanation and code

5. https://www.youtube.com/watch?v=firXjwZ_6KI
Video with pseudocode


![pseudocode](images/image.png)

![algo_1](<images/Screenshot 2025-10-24 at 5.26.24 PM.png>)

![algo_2](<images/Screenshot 2025-10-24 at 10.31.47 PM.png>)

![VLA_gen2](images/VLA-gen2.png)

Implementation in Pi0 algorithm.