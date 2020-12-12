# Journal

## 2020.12.12
1. 發現將原始Trainig data 放入模型內預測，答案是對的
> 推測可能模型overfitting了(也就是說做了太多次epoch)
2. 但如果將原始資料放進ptrtrained model預測，一樣會造成錯誤
> 學長推測是因為程式中沒有使用到 **model.eval()** 這個函式
