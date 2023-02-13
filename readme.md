#Tenten

Implements a version of the classic puzzle game 10 x 10.
The princible is simple and comparable to Tetris. 
As selection of three pieces and a board of with 10 times 10 squares are available:

```
Score: 38                                     |>| Score: 77 (+9 +10 +20)
     1  1  1  1  1  1  1  1  1  0  0  0       |>|      1  1  1  1  1  1  1  1  1  0  0  0 
     1  0  0  0  0  0  1  1  0  0  0  0       |>|      1  0  0  0  0  0  1  1  0  0  0  0   
     1  0  0  0  0  0  1  0  0  0  0  0       |>|      1  0  0  0  0  0  1  0  0  0  0  0 
     1  0  0  0  0  0  0  0  0  0  0  0       |>|      1  0  0  0  0  0  0  0  0  0  0  0
     1  0  0  0  0  0  0  0  0  0  0  0       |>|      1  0  0  0  0  0  0  0  0  0  0  0 
     1  0  0  0  0  0  0  0  0  0  0  0       |>|      1  0  0  0  0  0  0  0  0  0  0  0 
     0  0  0  0  0  0  0  0  0  0  0  0       |>|      0  0  0  0  0  0  0  0  0  0  0  0
     0  0  1  0  0  0  0  1  1  0  0  0 <     |>|      0  0  1  0  0  0  0  1  1  1  1  1  
     1  1  1  1  1  1  1  1  1  0  0  0 <     |>|    > 0  0  0  0  0  0  0  0  0  0  0  0 <
     1  1  1  1  1  1  1  1  1  0  0  0 <     |>|    > 0  0  0  0  0  0  0  0  0  0  0  0 <
                                ^  ^  ^       |>|
--------------------------------------------- |>|---------------------------------------------
1  0  0  0  0 |>1  1  1< 0  0 | 0  0  1  0  0 |>| 1  0  0  0  0 |               | 0  0  1  0  0  
1  0  0  0  0 |>1  1  1< 0  0 | 0  0  1  0  0 |>| 1  0  0  0  0 |               | 0  0  1  0  0  
1  0  0  0  0 |>1  1  1< 0  0 | 1  1  1  0  0 |>| 1  0  0  0  0 |               | 1  1  1  0  0  
1  0  0  0  0 | 0  0  0  0  0 | 0  0  0  0  0 |>| 1  0  0  0  0 |               | 0  0  0  0  0  
1  0  0  0  0 | 0  0  0  0  0 | 0  0  0  0  0 |>| 1  0  0  0  0 |               | 0  0  0  0  0  
```

Game implementation found under game. Playable by running ui.py (requires TKinter).

Setup so that ML algorithms can be setup on it.
Currently used for reinforcement learning Deep Q learning. Maybe extended in the future.
Builds on Pytorch and uses CUDA.