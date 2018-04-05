1-1-2:
goal: Train on Actual Tasks
1. 進入路徑：
cd HW1/HW1-1

2. 如果想重建畫出的結果圖(涵蓋訓練完成後的history資訊)：

\# it will call deep.pickle, mid.pickle and shallow.pickle to plot the loss and acc. <br>
python3 hw1_1_plot_img.py <br>

3. 如果想重新跑訓練部分執行以下指令：

\# training for deep model and save model history information as deep.pickle <br>
python3 hw1_1_deep.py <br>

\# training for middle model and save model history information as mid.pickle <br>
python3 hw1_1_mid.py <br>

\# training for shallow model and save model history information as shallow.pickle <br>
python3 hw1_1_shallow.py <br>

\# 跑完後會將keras裡模型history存成pickle檔
\# 透過2. 可以完成繪圖
______________________________________________________
1-2-2:
goal: Observe Gradient Norm During Training
1. 進入路徑：
cd HW1/HW1-2

2. 執行程式：

\#跑訓練過程，順便畫圖 <br>
python3 hw1_2_2.py <br>

1-2-bonus:
goal: Use any method to visualize the error surface(用起始跟終止權重內插的方法)
1. 進入路徑：
cd HW1/HW1-3

2. 直接重建結果，根據所存的train_loss.npy跟val_loss.npy畫出error surface：

\# plot error surface for training and validation losses <br>
python3 hw1_2_plot_bonus.py <br>

3. 如果要重新訓練，從這里開始。首先，訓練模型並save model(存權重):

\# training and saving the model <br>
python3 hw1_2_bonus_train.py <br>

4. 恢復模型權重來做線性內插再算loss，並將losses存成numpy array：

\# restore model(weights) and do the linear interpolation <br>
\# (sampling 2000 points between 0, 1., which is alpha value) <br>
\# (1-alpha)*initial weights + alpha*final weights <br>
\# it will save the training and validation losses <br>
python3 hw1_2_bonus.py <br>

5. 根據所存的train_loss.npy跟val_loss.npy畫出error surface：
\# 透過2. 可以完成繪圖
_________________________________________________

1-3-3-part 2:
goal: Flatness v.s. Generalization - part2
1. 進入路徑：
cd HW1/HW1-3

2. 訓練不同training approaches並畫圖
python3 hw1_3_3_2.py
__________________________________________________

