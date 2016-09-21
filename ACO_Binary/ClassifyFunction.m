function Out = ClassifyFunction(pop,nVar,Data)

    FeatIndex = pop; %Feature Index
    X1 = Data.X;% Features Set
    Y1 = Data.Y;% Class Information
    Holdout = 0.75;
    
    if pop==0
        FeatIndex = round(rand(1 , nVar));
    end
    X1 = X1(:,FeatIndex);
    
    [row,~] = size(X1);
    
    X1Train = X1(1:round(Holdout*row),:);
    Y1Train = Y1(1:round(Holdout*row),:);
    X1Test = X1(round(Holdout*row):end,:);
    Y1Test = Y1(round(Holdout*row):end,:);
    
     
    Out.FeatIndex=FeatIndex;
       
    Out
figure
plotconfusion(Out.target,Out.output);

function [p , r , fr , cm] = PrecisionRecall(target , output , x)


    
    numClass = length(unique(target));
    
    cm = confusionmat(target, output);
    
    for i = 1 : numClass
        tp = cm(i , i);
        precision(i) =  tp / sum(cm(: , i));%#ok
        recall(i) = tp / sum(cm(i , :));%#ok
    end
    
    p = mean(precision);
    r = mean(recall);
    fr=(size(x,2)-sum(x))/size(x,2);
    