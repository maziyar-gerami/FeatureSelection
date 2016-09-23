%% Start of Program

clc
clear
close all
global Data nVar

%% Problem Definition

Data  = load('Data1.mat');                       % Load Data

Data.X = NormData(Data);                        % Normalize Data

nVar=size(Data.X,2);                            % Number of Decision Variables

CostFunction=@(x)mRMR(x,nVar,Data);             % Cost Function

%% ACO Parameters

ACO_MaxIt=10;               % Maximum Number of Iterations

nAnt=50;              % Number of Ants (Population Size)

Q=1;

q0=0.7;             % Exploitation/Exploration Decition Factor

tau0=1;             % Initial Phromone

alpha=0.7;          % Phromone Exponential Weight
beta=0.3;           % Heuristic Exponential Weight

rho=0.7;           % Evaporation Rate

%% GA Parameters

GA_MaxIt=20;      % Maximum Number of Iterations

nPop=50;        % Population Size

pc=0.8;                 % Crossover Percentage
nc=2*round(pc*nPop/2);  % Number of Offsprings (Parnets)

pm=0.2;                 % Mutation Percentage
nm=round(pm*nPop);      % Number of Mutants

mu=0.2;         % Mutation Rate

%% Initialization

eta=ones(nVar,2);               % Heuristic Information Matrix

tau=tau0*ones(nVar, 2);         % Phromone Matrix

BestCost=zeros(ACO_MaxIt+GA_MaxIt,1);       % Array to Hold Best Cost Values

% Empty Ant
empty_pop.Position=[];
empty_pop.Features=[];
empty_pop.Cost=[];


% Ant Colony Matrix
ant=repmat(empty_pop,nAnt,1);

% Best Ant
BestAnt.Cost=+inf;

%% ACO Main Loop
tic
for i=1:ACO_MaxIt
    %Move Ants
    for k=1:nAnt
        
        for n=1:nVar
            
           q= rand;
           
            if(q<=q0)
                
                [~, idx] = max((tau(n,:)).^alpha.*(eta(n,:)).^beta);
                
            else
                
                P = (((tau(n,:)).^alpha.*(eta(n,:)).^beta)./sum((tau(n,:)).^alpha.*(eta(n,:)).^beta));
                
                P = P/sum(P);
                
                P = P';
                
                idx = RouletteWheelSelection(P);
                
            end
            
            ant(k).Position(n) = idx;
            
        end
        
        [ant(k).Cost,ant(k).Features]=mRMR(ant(k).Position, nVar, Data);
        
    end
    
    
    [~, SortOrder]=sort([ant.Cost], 'ascend');
    ant=ant(SortOrder);
    
    if ant(1).Cost<BestAnt.Cost
        BestAnt = ant(1);
    end

    update.Costs = NormAntCosts(ant, BestAnt);
    
    update.Ants = [BestAnt; ant ];
    
    % update best path
    for j=1:nVar    
                
                tau(j, update.Ants(1).Position(j))= tau(j, update.Ants(1).Position(j))+ rho*(update.Ants(1).Cost);
                
    end
    
    % update other paths
    for k=2:20
        
        for j=1:nVar    
                
                tau(j, update.Ants(k).Position(j))= tau(j, update.Ants(k).Position(j))+ update.Ants(k).Cost;
                
        end
        
    end
     
    % Evaporation
    tau=(1-rho)*tau;
    
    % Store Best Cost
    BestCost(i)=BestAnt.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(i) ': Best Cost = ' num2str(BestCost(i))]);
end

pop=repmat(empty_pop,nPop,1);


pop = ant;

for it=1:GA_MaxIt
    
    
    P=[pop.Cost]/sum([pop.Cost]);
    
    % Crossover
    popc=repmat(empty_pop,nc/2,2);
    for k=1:nc/2
        
        %  Select Parents Indices
        i1=RouletteWheelSelection(P);
        i2=RouletteWheelSelection(P);

        % Select Parents
        p1=pop(i1);
        p2=pop(i2);
        
        % Apply Crossover
        flag = true;
            
        [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position, Data);
            
        % Evaluate Offsprings
        popc(k,1).Features = find(popc(k,1).Position>1);
        popc(k,1).Cost = CostFunction(popc(k,1).Position);
        
        popc(k,2).Features = find(popc(k,2).Position>1);
        popc(k,2).Cost = CostFunction(popc(k,2).Position);
        
        
    end
    popc=popc(:);
    
    % Mutation
    popm=repmat(empty_pop,nm,1);
    for k=1:nm
        
        % Select Parent
        i=randi([1 nPop]);
        p=pop(i);
        
        % Apply Mutation    
    
        popm(k).Position=Mutate(p.Position,mu);
        popm(k).Features = find(popm(k).Position>1);
        
        % Evaluate Mutant
        popm(k).Cost=CostFunction(popm(k).Position);
        
    end
    
    % Create Merged Population
    pop=[pop
         popc
         popm];
    
    popGlobal = [pop; popc;popm]; 
    % Sort Population
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);
    
    
    % Truncation
    pop=pop(1:nPop);
    Costs=Costs(1:nPop);
    
    % Store Best Solution Ever Found
    BestSol=pop(1);
    
    % Store Best Cost Ever Found
    BestCost(ACO_MaxIt)=BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end



%% Results
figure;
plot (BestCost, 'LineWidth', 2);
xlabel ('Iteration');
ylabel ('Best Cost');
BestSol.Features
