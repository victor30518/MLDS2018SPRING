# 1-1-1 Simulate a funtion
1. 進入路徑：<br>
```Bash
cd code/1-1/1-1-1
```
2. 如果想重建畫出的結果圖(涵蓋訓練完成後的history資訊)，執行後圖片會存在根目錄：<br>
```Bash
python3 hw1_1_1_plot.py
```
3. 如果想重新跑訓練部分執行以下指令：
```Bash
# 注意!執行後將覆蓋掉原本的資料
python3 hw1_1_1_train.py
```
# 1-1-2 Train on Actual Tasks
1. 進入路徑：<br>
```Bash
cd code/1-1/1-1-2
```
2. 如果想重建畫出的結果圖(涵蓋訓練完成後的history資訊)：<br>
```Bash
# it will call deep.pickle, mid.pickle and shallow.pickle to plot the loss and acc.
python3 hw1_1_plot_img.py
```
3. 如果想重新跑訓練部分執行以下指令：<br>
```Bash
# training for deep model and save model history information as deep.pickle
python3 hw1_1_deep.py

# training for middle model and save model history information as mid.pickle
python3 hw1_1_mid.py

# training for shallow model and save model history information as shallow.pickle
python3 hw1_1_shallow.py

# 跑完後會將keras裡模型history存成pickle檔
```
4. 畫結果圖(涵蓋訓練完成後的history資訊)：<br>
```Bash
# 透過2. 可以完成繪圖
python3 hw1_1_plot_img.py
```
# 1-1-Bonus
1. 進入路徑：<br>
```Bash
cd code/1-1/bonus
```
2. 如果想重建畫出的結果圖(涵蓋訓練完成後的history資訊)：<br>
```Bash
python3 hw1_1_bonus_plot.py
```
______________________________________________________
# 1-2-1 Visualize the optimization process
1. 進入路徑：<br>
```Bash
cd code/1-2/1-2-1
```
2. 如果想重建畫出的結果圖，執行後圖片會存在根目錄：<br>
```Bash
python3 hw1_2_1_plot.py
```
3. 如果想重新跑訓練部分執行以下指令：
```Bash
# 注意!執行後將覆蓋掉原本的資料
python3 hw1_2_1_train.py
```
# 1-2-2 Observe Gradient Norm During Training
1. 進入路徑： <br>
```Bash
cd code/1-2/1-2-2
```
2. 執行程式： <br>
```Bash
#跑訓練過程，順便畫圖
python3 hw1_2_2.py
```
# 1-2-3 What happens when gradient is almost zero?
1. 進入路徑： <br>
```Bash
cd code/1-2/1-2-3
```
2. 執行程式： <br>
```Bash
#執行一次訓練過程 並且印出loss與minimal ratio
python3 hw1_2_3.py
```
# 1-2-bonus Use any method to visualize the error surface(用起始跟終止權重內插的方法)
1. 進入路徑： <br>
```Bash
cd code/1-2/1-2-2
```
2. 直接重建結果，根據所存的train_loss.npy跟val_loss.npy畫出error surface：<br>
```Bash
# plot error surface for training and validation losses
python3 hw1_2_plot_bonus.py
```
3. 如果要重新訓練，從這里開始。首先，訓練模型並save model(存權重):<br>
```Bash
# training and saving the model
python3 hw1_2_bonus_train.py
```
4. 恢復模型權重來做線性內插再算loss，並將losses存成numpy array：<br>
```Bash
# restore model(weights) and do the linear interpolation
# (sampling 2000 points between 0, 1., which is alpha value)
# (1-alpha)*initial weights + alpha*final weights
# it will save the training and validation losses
python3 hw1_2_bonus.py
```
5. 根據所存的train_loss.npy跟val_loss.npy畫出error surface：<br>
```Bash
# 透過2. 可以完成繪圖
python3 hw1_2_plot_bonus.py
```
_________________________________________________
# 1-3-1-Can network fit random variables?
1. 進入路徑：<br>
```Bash
cd code/1-3/1-3-1
```
2. 訓練model來fit random label的data，並且產生圖形：<br>
```Bash
python3 hw1_3_1.py
```
# 1-3-2-Number of parameters v.s. Generalization
1. 進入路徑：<br>
```Bash
cd code/1-3/1-3-2
```
2. 執行一次訓練並且繪圖（改變變數neuron的值，可增減參數量來進行訓練）：<br>
```Bash
python3 hw1_3_1.py
```
# 1-3-3-part1 Flatness v.s. Generalization - part1
1. 進入路徑：<br>
```Bash
cd code/1-3/1-3-3-1
```
2. 如果想重建畫出的結果圖，執行後圖片會存在根目錄：<br>
```Bash
python3 hw1_3_3_1_plot.py
```
3. 如果想重新跑訓練部分執行以下指令：
```Bash
# 注意!執行後將覆蓋掉原本的資料
python3 hw1_3_3_1_train_1.py
python3 hw1_3_3_1_train_2.py
```
# 1-3-3-part2 Flatness v.s. Generalization - part2
1. 進入路徑：<br>
```Bash
cd code/1-3/1-3-3-2
```
2. 訓練不同training approaches並畫圖<br>
```Bash
python3 hw1_3_3_2.py
```
__________________________________________________

