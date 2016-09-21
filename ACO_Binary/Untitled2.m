load fisheriris
xdata = meas(51:end,3:4);
group = species(51:end);
figure;
svmStruct = svmtrain(xdata,group,'ShowPlot',true)

a = svmclassify(svmStruct, meas(1:50,3:4));