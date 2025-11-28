base_dir = pwd;
addpath(genpath(fullfile(base_dir,'helper_functions')))
addpath(fullfile(base_dir,'extrapolation_data'))

load best_results_across_batchsizes.mat

for i = 1:14
    ind = (i-1)*4+(1:4);
    m_best_ce(i) = mean(best_ce(ind));
    m_best_dim(i) = mean(best_dim(ind));
    ce_res{i} = best_ce(ind);
    dim_res{i} = best_dim(ind);
end

% % plot cross entropy (as a measure of performance itself, one can see
% % things have saturated)
% figure
% h = plotSpread(ce_res);
% set(h{1},'MarkerSize',20,'MarkerFaceColor',[0 0 0],'MarkerEdgeColor',[0 0 0])
% hold on

% plot dimensionality across dataset sizes
figure('Position',[450 1 1200 900]);
h = plotSpread(dim_res);
set(h{1},'MarkerSize',20,'MarkerFaceColor',[0 0 0],'MarkerEdgeColor',[0 0 0])
hold on

% extrapolate function using an exponential in the shape of
% a + b exp(-cx)

% parameter A: when dataset size is infinite and assuming a function that decays,
% this parameter reflects the dimensionality in the limit. Thus, this number 
% must be positive and larger than the current dimensionality
% (let's assume 100 = parameter of 1)
% parameter B: estimated to be negative (no growth but decay)
% parameter C: no clear estimate, so let's use 1
A = expcurvefit(1:14,m_best_dim/100,[1 -0.5 1]);
Aest = A(1); Best = A(2); Cest = A(3);
Fitted = Aest + (Best.* exp(-Cest * [1:60 100]));
plot(100*Fitted(1:60),'Color',[0 0 0],'LineWidth',2)
xlim([0 61])

%% bootstrap

% we want to find the error of our curve fitting exercise

rng(1)
if ~exist('Fitted_boot.mat','file')
    
    for i = 1:14
        ind = (i-1)*4+(1:4);
        for j = 1:1000
            ind2 = ind(randi(4,4,1));
            me_best_dim_boot(i,j) = mean(best_dim(ind2));
        end
    end
    
    for j = 1000:-1:1
        Aboot(:,j) = expcurvefit(1:14,me_best_dim_boot(:,j)'/100,[0.675 -0.67 0.09]);
    end
    
    for j = 1000:-1:1
        Fitted_boot(:,j) = Aboot(1,j) + (Aboot(2,j).* exp(-Aboot(3,j) * [1:60 100]));
    end
    
    save('Fitted_boot.mat','Fitted_boot')
    
else
    load('Fitted_boot.mat')
end

plot(prctile(100*Fitted_boot(1:60,:)',2.5),'Color',[0.5 0.5 0.5])
plot(prctile(100*Fitted_boot(1:60,:)',100-2.5),'Color',[0.5 0.5 0.5])


% Now plot all other 71 models (get their dimensionality)
if exist('model_dims.mat','file')
    load('model_dims.mat')
else
    refdir = fullfile('..','models');
    for i_model = 1:72
        if i_model == 64
            continue
        end
        fn = dir(fullfile(refdir,sprintf('seed%02i',i_model-1),'0.00385','*.txt'));
        fn = fullfile(fn(end).folder,fn(end).name);
        tmp = load(fn);
        % remove empty dimensions
        tmp = tmp';
        tmp2 = tmp(:,any(tmp>0.1));
        n_dim_reference(i_model) = size(tmp2,2);
    end
    n_dim_reference(64) = [];
    save('model_dims.mat','n_dim_reference')
end
rng(1)
y = min(n_dim_reference):max(n_dim_reference);
d = histc(n_dim_reference,y);
h2 = distributionPlot({[y' d']},'xValues',4578093/1e5,'showMM',0,'distWidth',2);
set(h2{1},'EdgeColor',[0 0 0],'FaceColor',[1 1 1],'LineWidth',1)

% Now add marker where new dataset size is (later manually replace with
% star)
xpos = 4578093/1e5;
plot(xpos,66,'o','markerfacecolor',[0 0 0],'markersize',6)

ax = gca;
ax.XTick = 2:2:60;
str = num2cell(ax.XTick/10);
ax.XTickLabel = str;
ax.YTick = 0:5:80;
ylabel('Number of dimensions')
xlabel('Dataset size (in million trials)')

set(gcf,'Renderer','painters')

print(gcf,'-dpdf','temp1.pdf','-bestfit')