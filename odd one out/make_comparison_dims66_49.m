base_dir = pwd;
data_dir = fullfile(base_dir,'data');
variable_dir = fullfile(base_dir,'variables');
helper_dir = fullfile(base_dir,'helper_functions');

load(fullfile(data_dir,'spose_embedding_66d_sorted.txt'))
load(fullfile(data_dir,'embedding49','spose_embedding_49d_sorted.txt'))

load(fullfile(variable_dir,'labels_short.mat'));
labels66 = labels_short;

load(fullfile(data_dir,'embedding49','labels_short49.mat'));
labels49 = labels_short;
labels49{17} = 'coarse pattern';


clear labels

figure, imagesc(corr(spose_embedding_49d_sorted,spose_embedding_66d_sorted));
ax = gca;
ax.XTick = 1:66;
ax.XTickLabels = labels66;
ax.XTickLabelRotation = 90;
ax.YTick = 1:49;
ax.YTickLabels = labels49;

% threshold the dimensions similarity matrix
% then figure out which dimensions already existed and check if they
% already showed a correlation, if they did then see if the remaining
% correlation is just mirroring that and if yes remove that correlation

% perhaps we just sort dimensions in a way that maximizes the correlation,
% i.e. to make it as diagonal as possible (also removing the
% non-reproducible dims)


c = corr(spose_embedding_49d_sorted,spose_embedding_66d_sorted);

% now let's sort dimensions by maximal cross-correlation
[~,ii] = max(c);
% remove ones that have already appeared
for i = 2:66
    if any(ii(1:i-1)==ii(i))
        ii(i) = 100;
    end
end
[~,si] = sort(ii);

% now remove original cross-correlation matrix to find differences
c_base = [(corr(spose_embedding_49d_sorted) - eye(49)) zeros(49,17)];
c_adapted = c(:,si)-c_base;
zeroind = c_adapted(:,50:66)>0.3;
zeroind = [c_adapted(1:49,1:49) zeroind]<0.3;
c_adapted(zeroind) = 0;

% dimension reproducibility: sorting dimensions by maximal correlation
disp(sort(diag(c_adapted),'descend'))
% only three dimensions not reproduced: handicraft-related and cylindrical
% (sparsest) as well as partial reproduction of dim 32 (flat/patterned),
% which was among the least interpretable dims but which originally was
% quite reproducible

figure, imagesc(c(:,si))
ax = gca;

ax.YTick = 1:49;
ax.YTickLabels = labels49;
ax.XTick = 1:66;
ax.XTickLabels = labels66(si);
ax.XTickLabelRotation = 90;

clear x
[x(:,1) x(:,2)] = ind2sub([49 66],find(c_adapted));
% x(:,2) = si(x(:,2));
x = sortrows(x,[1 2]);

%% now extend x to get the size of each one right, and make two more variables to complete the first where variables 
%% without a pair from the other side will be plotted as flat until the middle
n_rep49 = max(histc(x(:,1),1:49));
n_rep66 = max(histc(x(:,2),1:66));
n_rep_total = n_rep49*n_rep66; % the product of both is the safest


%% plot flow diagram

addpath(fullfile(helper_dir,'external','patchline'))

% start out with a version where we map 49 points to 66 points

pos_x49 = -8;
pos_x66 = 8;
pos_y49 = linspace(1,100,49);
pos_y66 = linspace(1,100,66);

% flow components as sigmoid function from one point to another with fixed
% slope

% given a known starting point and end point, what is this function?
% a is starting point and end point is a+b, so to add start and end point
% in function this should do the job
sigmoid = @(x,a,b) a+ (b-a)./(1+exp(-x));

k = linspace(pos_x49,pos_x66,100);

load(fullfile(data_dir,'embedding49','colors49.mat'))

figure('Position',[300 1 1200 1200])
hold on
rng(42) % was 4
for i = 1:size(x,1)
    if x(i,1)==x(i,2)
        hl(i) = patchline([k(1) k(end)],[x(i,1) x(i,2)],'edgecolor',colors49(x(i,1),:),'linewidth',8*c_adapted(x(i,1),x(i,2)),'edgealpha',0.3);
    else
    ktmp = linspace(pos_x49+4*randn,pos_x66+4*randn,100);
    ytmp = sigmoid(ktmp,(x(i,1)),(x(i,2)));
    while abs(ytmp(1)-x(i,1))>0.1 || abs(ytmp(end)-x(i,2))>0.1
        ktmp = linspace(pos_x49+2*randn,pos_x66+2*randn,100);
        ytmp = sigmoid(ktmp*randn,(x(i,1)),(x(i,2)));
    end
    hl(i) = patchline(k,ytmp,'edgecolor',colors49(x(i,1),:),'linewidth',8*c_adapted(x(i,1),x(i,2)),'edgealpha',0.7);
    end
end
yl = ylim;

a = gca;
a.XTick = [];
drawnow
a.XRuler.Axle.Visible = 'off'; % a is axis handle
a.YRuler.Axle.Visible = 'off';

set(gca,'YDir','reverse')
set(gca,'YTick',1:49,'YTickLabel',labels49)
yyaxis right
ylim(yl)
set(gca,'YDir','reverse')
set(gca,'YTick',1:66,'YTickLabel',labels66(si).')

set(gcf,'Units','centimeters')
screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)])
set(gcf,'PaperSize',screenposition(3:4))
doprint = 0;
if doprint
print(gcf,'-dpdf','temp1.pdf','-bestfit') %#ok<UNRCH> 
end