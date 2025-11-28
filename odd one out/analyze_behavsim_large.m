% This script runs all relevant analyses included for the behavioral 
% dataset of THINGS-data (Hebart et al., submitted) with the new dataset 
% of 4.7 million triplets and the 66d embedding. Many results computed here
% are not reported in the final manuscript but are kept for completeness.

% run this script from where it is located
base_dir = pwd;
data_dir = fullfile(base_dir,'data');
dataset_dir = fullfile(data_dir,'triplet_dataset');
variable_dir = fullfile(base_dir,'variables');

%% Add relevant toolboxes

% t-SNE from: https://lvdmaaten.github.io/tsne/#implementations
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load relevant data

% load embedding
spose_embedding66 = load(fullfile(data_dir,'spose_embedding_66d_sorted.txt'));
% get dot product (i.e. proximity)
dot_product66 = spose_embedding66*spose_embedding66';
% load similarity computed from embedding (using embedding2sim.m)
load(fullfile(data_dir,'spose_similarity.mat'))
dissim = 1-spose_sim;
% load 10% validation data
triplet_validationdata66 = load(fullfile(dataset_dir,'validationset.txt'))+1; % 0 index -> 1 index
% load test sets

% load 48 object set, also get their indices
load(fullfile(data_dir,'RDM48_triplet.mat'))
load(fullfile(data_dir,'RDM48_triplet_splithalf.mat'))
% load typicality ratings for 27 categories
load(fullfile(data_dir,'typicality_data27.mat'))
% load answers from participants labeling the images
load(fullfile(data_dir,'dimlabel_answers.mat'))
load(fullfile(data_dir,'dimension_ratings.mat'))

%% get dimension labels, short labels and colors

load(fullfile(variable_dir,'labels.mat'))
load(fullfile(variable_dir,'labels_short.mat'))

h = fopen(fullfile(variable_dir,'colors.txt')); % get list of colors in hexadecimal format

col = zeros(0,3);
while 1
    l = fgetl(h);
    if l == -1, break, end
    
    col(end+1,:) = reshape(sscanf(l(2:end).','%2x'),3,[]).'/255; % hex2rgb
    
end
fclose(h);

col(1,:) = [];
col([1 2 3],:) = col([2 3 1],:);

% now adapt colors
colors = col([1 20 3 38 9 7 62 57 13 6 24 25 50 48 36 53 46 28 62 18 15 58 2 11 40 45 27 55 36 30 34 31 41 16 27 61 17 36 57 25 63],:); colors(end+1:49,:) = col(8:56-length(colors),:);
colors(46,:) = colors(46,:)-0.2; % medicine related is too bright, needs to be darker

colors = colors([1 2 3 4 6 5 12 8 10 9 13 11 7 15 18 14 16 19 21 17 22 33 17 23 20 27 26 19 24 37 20 28 47 31 39 30 36 43 29 35 38 9 6 25 49 40 42 37 44 25 41 12 20 45 7 41 46 2 23 34 5 33 13 31 40 32],:);

colors([20 28 30 31 41 42 43 45 50 52 53 55 56 58 59 61 62 63 64 65],:) = 1/255*...
    [[146 78 167];
    [143 141 58];
    [255 109 246];
    [71 145 205];
    [0 118 133];
    [204 186 45];
    [0 222 0];
    [222 222 0];
    [100 100 100];
    [40 40 40];
    [126 39 119];
    [177 177 0];
    [50 50 150];
    [120 120 50];
    [250 150 30];
    [40 40 40];
    [220 220 220];
    [90 170 220];
    [140 205 150];
    [40 170 225]];

clear col h l


%% Load smaller version of images, words, and unique IDs for each image

load(fullfile(variable_dir,'sortind.mat'));
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'unique_id.mat'))
load(fullfile(variable_dir,'words48.mat'))

% now download images if not already done
if ~exist(fullfile(variable_dir,'images1854.mat'),'file')
    disp('1854 preview images not found, downloading...')
    disp('(this is done only the first time this code is called)')
    websave(fullfile(variable_dir,'images1854.mat'),'https://osf.io/k4cyq/download');
    load(fullfile(variable_dir,'images1854.mat'))
    % bring in correct format
    for i = 1:1854
        images1854(i).im = im{sortind(i)}; % sortind to get incorrect order correct
        images1854(i).name = unique_id{i};
    end    
    save(fullfile(variable_dir,'images1854.mat'),'images1854')
    clear im imwords
    disp('done.')
else
    load(fullfile(variable_dir,'images1854.mat'))
end

[~,~,wordposition48] = intersect(words48,words,'stable');

%% Get embedding and relevant vectors

% Get similarity plot
ind = clustering_algorithm(3,5,spose_embedding66); % somewhat arbitrary way of sorting objects, according to 3 most dominant dimensions in each object
hf = figure; hf.Position(3:4) = [600 600]; imagesc(spose_sim(ind(1:10:end),ind(1:10:end)),[0 0.9])
colormap(viridis)
axis equal off
drawnow

%% Predict behavior and similarity

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_validationdata66),1);
behav_predict_prob = zeros(length(triplet_validationdata66),1);
rng(42) % for reproducibility
for i = 1:length(triplet_validationdata66)
    sim(1) = dot_product66(triplet_validationdata66(i,1),triplet_validationdata66(i,2));
    sim(2) = dot_product66(triplet_validationdata66(i,1),triplet_validationdata66(i,3));
    sim(3) = dot_product66(triplet_validationdata66(i,2),triplet_validationdata66(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc = 100*mean(behav_predict==1);
% get prediction for each object
for i_obj = 1854:-1:1
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_validationdata66==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_validationdata66==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95 = 1.96*std(behav_predict_obj)/sqrt(1854);


%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%
testset1 = load(fullfile(dataset_dir,'testset1.txt'));
testset1 = testset1+1;
% compute noise ceiling (since data are already sorted with the choice in 
% the last column, we just add a column of 3 indicating which one is the
% choice
[nc1,nc1_ci95] = get_noiseceiling([testset1 3*ones(size(testset1,1),1)]);


testset2 = load(fullfile(dataset_dir,'testset2.txt'));
testset2 = testset2+1;
[nc2,nc2_ci95] = get_noiseceiling([testset2 3*ones(size(testset2,1),1)]);


testset3 = load(fullfile(dataset_dir,'testset3.txt'));
testset3 = testset3+1;
[nc3,nc3_ci95] = get_noiseceiling([testset3 3*ones(size(testset3,1),1)]);

testset2_repeated = load(fullfile(dataset_dir,'testset2_repeat.txt'));
testset2_repeated = testset2_repeated+1;
% where are the choices in these two datasets the same for each triplet type separately?
% first sort each triplet to find all triplets that are the same type
% sort each triplet and change choice id
testset2_repeated_sorted = sort(testset2_repeated,2);
NCstr = num2cell(num2str(testset2_repeated_sorted),2);
uid = unique(NCstr);
nid = length(uid);
for i = 1:nid
    ind = strcmp(NCstr,uid{i});
    consistency_within(i,1) = mean(testset2(ind,3)==testset2_repeated(ind,3)); % the best one divided by all
end
nc2_within = mean(consistency_within)*100;
nc2_within_ci95 = 1.96 * std(consistency_within)*100 / sqrt(nid);

% NB: we don't use testset2 repeated for computing a regular noise ceiling
% (even though we could) since it is biased: the previous choice could have
% affected the choice in this one. However, empirically the result is
% comparable

%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%
hf = figure;
hf.Position(3:4) = [900 1200];
% first plot noise ceiling
wid = 8;
x = 1+ [-wid wid wid -wid];
nc_u = nc1+nc1_ci95;
nc_l = nc1-nc1_ci95;
y = [nc_l nc_l nc_u nc_u];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
hold on
% now plot results
% ha1 = plot(1,behav_predict_train_acc,'o','MarkerFaceColor',[1 0 0],'MarkerEdgeColor','none','MarkerSize',12);
% ha2 = plot(2,behav_predict_acc,'o','MarkerFaceColor',[0 0 0],'MarkerEdgeColor','none','MarkerSize',12);
ha3 = bar(1,behav_predict_acc,'FaceColor',[0 0 0],'EdgeColor','none','BarWidth',6);
hb = errorbar(6,behav_predict_acc,behav_predict_acc_ci95,'Color',[0 0 0],'LineWidth',3);
hb = plot(x(1:2),[33.3333 33.3333],'r','LineWidth',3);
axis equal
xlim(x(1:2))
ylim([30 75])

hax = gca;
hax.TickDir = 'both';
hax.XTick = [];
hax.XColor = [0 0 0];
hax.YColor = [0 0 0];
hax.LineWidth = 1;
hax.Box = 'off';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now compare similarity from model to similarity in behavior %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for that focus on those 48 objects only rather than the entire matrix
esim = exp(dot_product66);
cp = zeros(1854,1854);
ctmp = zeros(1,1854);
for i = 1:1854
    for j = i+1:1854
        ctmp = zeros(1,1854);
        for k_ind = 1:length(wordposition48)
            k = wordposition48(k_ind);
            if k == i || k == j, continue, end
            ctmp(k) = esim(i,j) / ( esim(i,j) + esim(i,k) + esim(j,k) );
        end
        cp(i,j) = sum(ctmp); % run sum first, divide all by 48 later
    end
end
cp = cp/48; % complete the mean
cp = cp+cp'; % symmetric
cp(logical(eye(size(cp)))) = 1;

spose_sim48 = cp(wordposition48,wordposition48);

% compare to "true" similarity
r48 = corr(squareformq(spose_sim48),squareformq(1-RDM48_triplet));

% run 1000 bootstrap samples for confidence intervals, bootstrap cannot 
% be done across objects because otherwise it's biased
rng(2)
rnd = randi(nchoosek(48,2),nchoosek(48,2),1000);
c1 = squareformq(spose_sim48);
c2 = squareformq(1-RDM48_triplet);
for i = 1:1000
    r48_boot(:,i) = corr(c1(rnd(:,i)),c2(rnd(:,i)));
end
r48_ci95_lower = tanh(atanh(r48) - 1.96*std(atanh(r48_boot),[],2)); % reflects 95% CI
r48_ci95_upper = tanh(atanh(r48) + 1.96*std(atanh(r48_boot),[],2)); % reflects 95% CI


h = figure;
h.Position(3:4) = [1200 768];
ha = subtightplot(1,3,1);
imagesc(spose_sim48,[0 1])
colormap(viridis)
hold on
text(24,-4,'predicted similarity matrix','HorizontalAlignment','center','FontSize',16);
axis off square tight
hb = subtightplot(1,3,2);
imagesc(1-RDM48_triplet,[0 1])
text(24,-4,'measured similarity matrix','HorizontalAlignment','center','FontSize',16);
colormap(viridis)
axis off square tight
hc = subtightplot(1,3,3);
hc.Position(1) = hc.Position(1)+0.02;
% plot(squareformq(spose_sim48),squareformq(1-RDM48_triplet),'o','MarkerFaceColor',[0.1651 0.4674 0.5581],'MarkerEdgeColor','none','MarkerSize',3);
plot(squareformq(spose_sim48),squareformq(1-RDM48_triplet),'o','MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor','none','MarkerSize',3);
hold on
plot([0 1],[0 1],'k')
axis square tight
xlabel('predicted similarity')
ylabel('measured similarity')
legend('R = 0.90','Location','NorthWest')

% get reliability of each split
reliability48 = corr(squareformq(1-RDM48_triplet_split1),squareformq(1-RDM48_triplet_split2));
% carry out Spearman-Brown correction
noise_ceiling48 = (2*reliability48)/(1+reliability48);

% get amount of variance explained (in the correlation or in behavior?)
variance_explained48 = r48.^2 / noise_ceiling48.^2;

% get stats
fprintf('Accuracy on validation data: %2.2f (95%% CI across objects: %2.2f)\n',mean(behav_predict_obj),behav_predict_acc_ci95) 
fprintf('Noise ceilings: \nNC1: %2.2f (95%% CI across objects: %2.2f),\nNC2: %2.2f (95%% CI: %2.2f), \nNC3: %2.2f (95%% CI: %2.2f), \nNC2_repeat: %2.2f (95%% CI: %2.2f)\n',nc1,nc1_ci95,nc2,nc2_ci95,nc3,nc3_ci95,nc2_within,nc2_within_ci95)
fprintf('Percent performance achieved (subtracting chance): between %2.2f and %2.2f for all three noise ceilings\n',100*(mean(behav_predict_obj)-100/3)/(max([nc1,nc2,nc3])-100/3),100*(mean(behav_predict_obj)-100/3)/(min([nc1,nc2,nc3])-100/3))

%% Show examples (loading high quality images)

% To use high quality images, see original code below (commented out)

% sort each dimension
[~,dimsortind] = sort(spose_embedding66,'descend');

dosave = 0;
n_im = 8;
for i_dim = [1:9 10:5:66] % selected dimensions
    
    figure('Position',[870 2043 2476 310],'color','none')
    
    % plot label
    subtightplot(1,9,1,0.005)
    ha = text(0,0,labels(i_dim));
    ha.Color = colors(i_dim,:);
    ha.FontSize = 16;
    ha.HorizontalAlignment = 'center';
    ylim([-1 1])
    xlim([-5 5])
    
    for i = 1:n_im
        
        subtightplot(1,9,i+1,0.005)
        imagesc(im{dimsortind(i,i_dim)})
        axis off square
        
    end
    
    if dosave
        hf = gcf;
        hf.Renderer = 'painters';
        % print(sprintf('tempim_%02i.pdf',i_dim),'-dpdf','-bestfit')
        saveas(hf,sprintf('tempim_%02i.svg',i_dim))
    end
    
end


%% Figure 4: Make crystal plot, with an example plot for abacus

dosave = 0;

% First, get 2d MDS solution
rng(42) % use fixed random number generator
[Y2,stress] = mdscale(dissim,2,'criterion','metricstress');

% Next, to visualize how tsne is run, we set clusters according to the
% strongest dimension in an object
[~,clustid] = max(spose_embedding66,[],2);

% Then, based on this solution, initialize t-sne solution with multiple
% perplexities in parallel (multiscale)
rng(1)
perplexity1 = 5; perplexity2 = 30;
D = dissim / max(dissim(:));
P = 1/2 * (d2p(D, perplexity1, 1e-5) + d2p(D, perplexity2, 1e-5)); % convert distance to affinity matrix using perplexity
figure
colormap(colors)
Ytsne = tsne_p(P,clustid,Y2);

% if interested plot words
figure
text(Ytsne(:,1),Ytsne(:,2),words)
xlim(1.05*[min(Ytsne(:,1)) max(Ytsne(:,1))])
ylim(1.05*[min(Ytsne(:,2)) max(Ytsne(:,2))])

% points within a specific polygon
% ff = [28;29;36;39;46;51;53;60;72;79;84;93;94;99;100;119;160;172;180;190;192;194;204;208;217;222;224;252;259;261;286;300;329;337;340;355;357;358;364;388;394;412;432;433;445;448;454;464;468;470;474;475;487;509;510;511;521;524;529;549;554;567;568;584;593;603;607;614;617;630;631;635;638;644;653;654;666;714;722;728;730;742;747;749;752;755;760;780;785;788;794;795;800;801;823;830;838;859;870;872;880;882;883;884;887;888;893;894;897;900;901;908;920;926;941;954;955;957;963;973;982;997;1020;1030;1047;1048;1049;1056;1063;1068;1096;1137;1139;1142;1143;1144;1145;1148;1156;1167;1180;1192;1202;1205;1208;1211;1214;1216;1219;1223;1228;1238;1246;1253;1256;1269;1278;1282;1287;1292;1305;1306;1312;1315;1325;1333;1336;1338;1342;1367;1370;1371;1375;1380;1382;1388;1391;1392;1400;1413;1414;1418;1425;1426;1433;1442;1462;1468;1489;1495;1507;1509;1511;1520;1534;1539;1540;1568;1572;1590;1603;1624;1628;1640;1643;1675;1677;1678;1713;1714;1720;1729;1735;1740;1766;1768;1781;1810;1812;1830;1836;1842;1848];
% ff = [4;14;15;31;32;40;41;43;44;64;65;76;82;103;113;115;118;122;134;136;150;157;165;174;202;207;228;233;236;264;278;282;289;292;317;326;338;348;349;350;366;367;376;379;413;417;418;422;431;458;471;497;547;551;552;578;582;619;620;646;659;674;697;699;702;703;704;705;706;713;720;774;835;839;861;877;905;907;911;915;921;931;952;967;975;990;995;1005;1008;1014;1023;1038;1041;1067;1075;1076;1078;1079;1080;1082;1099;1103;1112;1123;1129;1130;1132;1134;1135;1136;1151;1152;1157;1159;1168;1181;1182;1183;1184;1189;1195;1204;1210;1220;1225;1232;1236;1237;1245;1248;1254;1264;1275;1281;1285;1308;1324;1335;1349;1373;1402;1406;1419;1420;1498;1516;1518;1527;1541;1548;1558;1571;1585;1596;1619;1645;1651;1673;1709;1727;1733;1755;1779;1794;1796;1803;1806;1840;1851;1854];
% ff = [5;74;98;128;199;241;245;247;248;249;280;281;284;294;309;334;395;396;403;406;408;439;478;483;492;494;525;587;611;612;641;753;767;770;773;833;898;927;928;958;996;999;1000;1001;1012;1025;1028;1072;1089;1090;1128;1138;1162;1164;1203;1218;1241;1243;1301;1303;1337;1376;1389;1404;1408;1438;1447;1464;1550;1565;1611;1625;1642;1649;1667;1668;1692;1706;1715;1742;1751;1758;1762;1763;1772;1785;1792;1800];
ff = [14;15;32;40;41;43;64;65;82;103;113;118;136;157;165;202;207;228;233;236;264;282;289;292;338;348;367;413;417;458;551;578;659;674;697;703;705;706;713;720;774;835;839;861;877;905;907;911;921;931;975;1005;1008;1023;1038;1041;1067;1075;1076;1080;1103;1112;1123;1129;1134;1136;1152;1157;1168;1181;1182;1183;1189;1195;1248;1275;1308;1349;1373;1406;1516;1527;1541;1548;1558;1596;1645;1709;1733;1755;1803;1806;1854];

% Now add the "crystals", i.e. rose plots

v = zeros(200000,2);
f = zeros(100000,3);
ct = 0;
cnt1 = 0;
cnt2 = 0;

scaling = 2.8;

for ii = 1:1854
    if ii == 1854, fprintf('\n'), end
    [x,y] = pol2cart(repmat(linspace(0,2*pi,66+1),[66 1]),scaling*repmat(spose_embedding66(ii,:)',[1 66+1]));
    for i = 1:66
        ct = ct+1;
        v(cnt1+1:cnt1+3,:) = [Ytsne(ii,1) Ytsne(ii,2); x(i,i)+Ytsne(ii,1) y(i,i)+Ytsne(ii,2); x(i,i+1)+Ytsne(ii,1) y(i,i+1)+Ytsne(ii,2)];
        f(cnt2+1,:) = ((ct-1)*3 + (1:3));
        cnt1 = cnt1+3;
        cnt2 = cnt2+1;
    end
end
v(cnt1+1:end,:) = [];
f(cnt2+1:end,:) = [];

hf = figure;
hf.Position = [-393        1421         946         905];
patch('faces',f,'vertices',v,'FaceVertexCData',repmat(linspace(0,1,66),1,1854)','FaceColor','flat','edgecolor','none','facealpha',0.85)
colormap(colors)

axis equal off tight

%% classification analysis (no figure involved but included for reproducibility)

% results printed to screen (results even 1.26% better than for the original dataset)
predict_category(base_dir);

%% Now plot one example where we have zoomed in (microscope, word = 1000; bottle, word = 171; squid, word = 1529)

for i_example = [2 171 351 601 745 898 923 1000 1062 1131 1166 1198 1259 1284 1321 1529 1577 1787]
    
    
    v0 = [];
    f0 = [];
    v1 = [];
    ct = 0;
    [x0,y0] = pol2cart(repmat(linspace(0,2*pi,66+1),[66 1]),scaling*repmat(spose_embedding66(i_example,:)',[1 66+1]));
    [th,r] = cart2pol(x0,y0); [x1,y1] = pol2cart(th,r-0.05);
    for i = 1:66
        ct = ct+1;
        v0 = [v0; [0 0; x0(i,i) y0(i,i); x0(i,i+1) y0(i,i+1)]];
        f0 = [f0; ((ct-1)*3 + (1:3))];
        v1 = [v1; [0 0; x1(i,i) y1(i,i); x1(i,i+1) y1(i,i+1)]];
    end
    
    figure('Position',[1 103 1200 1200])
    patch('faces',f0,'vertices',v0,'FaceVertexCData',linspace(0,1,66)','FaceColor','flat','edgecolor','none','facealpha',0.5)
    colormap(colors)
    axis off square equal tight
    
    
    hold on
    clear ht rd
    rot = linspace(0,2*pi,66+1);
    rot = conv(rot,[0.5 0.5]);
    rot = rot(2:end-1);
    for i = 1:66
        if r(i,1)<1.5, continue, end
        vind = 3*(i-1);
        ht(i) = text(mean(v1(vind+(2:3),1)),mean(v1(vind+(2:3),2)),labels_short{i});
        ht(i).Rotation = rad2deg(rot(i));
        rd(i) = mod(rad2deg(rot(i)),180);
        ht(i).FontSize = 18; % was 18
        ht(i).FontName = 'Avenir Next';
        ht(i).HorizontalAlignment = 'right';
        if ht(i).Rotation>90 && ht(i).Rotation<270
            ht(i).Rotation = ht(i).Rotation+180;
            ht(i).HorizontalAlignment = 'left';
        end
    end
    title(words{i_example})
    
end


%% Predictions of human typicality

dosave = 0;

% Explanation of relevant variables
% categories27: category names for the 27 categories (alphabetically sorted)
% category27_typicality_rating_normed: typicality ratings for objects of the 27 categories (normed within each participant to make scale use more comparable)
% category27_ind: which of the 1,854 objects belong to each of the 27 categories
% category27_subind: which of the 27 categories do we use
% best_match27: which of the 66 dimensions best matches to the 27 categories (if any)

category27_subind = [1 3 4 6 7 8 9 10 12 14 17 18 21 22 23 24 26 27];
best_match27 = [3 NaN 30 4 NaN 49 58 18 11 2 NaN 6 NaN 40 NaN NaN 57 32 NaN NaN 5 16 17 54 NaN 8 26];

% to show relationship between categories and labels
% [categories27(category27_subind); labels(best_match27(category27_subind))]';


% extract relevant parts of embedding and typicalities
for i = 1:length(category27_subind)
    typicality_normed{i} = category27_typicality_rating_normed{category27_subind(i)};
    spose{i} = spose_embedding66(category27_ind{category27_subind(i)},best_match27(category27_subind(i)));
end

for i = 1:length(category27_subind)
    % using the unsorted one for both makes it easiest
    [r_typicality_s(i,1),p_typicality_s(i,1)] = corr(category27_typicality_rating_normed{category27_subind(i)}, spose_embedding66(category27_ind{category27_subind(i)},best_match27(category27_subind(i))),'tail','right','type','spearman');
end

[~,~,~,p_typicality_s_adjusted] = fdr_bh(p_typicality_s);
    
% Get typicality colors
typicality_colors = best_match27(category27_subind);
% sort typicality by size of correlation
[~,sortindtmp] = sort(r_typicality_s,'descend');
spose = spose(sortindtmp);
typicality_normed = typicality_normed(sortindtmp);
typicality_colors = typicality_colors(sortindtmp);

hf = figure('Position',[578 426 1674 703]);
clear ht hx hy
for i = 1:18
    subtightplot(4,5,i,0.05)
    if i <= 18
        plot(spose{i},typicality_normed{i},'o','MarkerFaceColor',colors(typicality_colors(i),:),'MarkerEdgeColor','none','MarkerSize',8)
%     else
%         plot(spose{i},typicality_normed{i},'o','MarkerFaceColor',[0.2 0.2 0.9],'MarkerEdgeColor','none','MarkerSize',8)
    end
    
    deltax = range(spose{i}); deltay = range(typicality_normed{i});
%     text(mean(spose{i}),mean(typicality_normed{i}),sprintf('%.2f',corr(spose{i}',typicality_normed{i}','type','spearman')))
    ht(i) = text(max(spose{i})-0.05*deltax,min(typicality_normed{i}+0.1*deltax),sprintf('%s = %.2f','\rho',r_typicality_s(sortindtmp(i))),'HorizontalAlignment','right');
    xlim([min(spose{i})-0.08*deltax max(spose{i})+0.08*deltax])
    ylim([min(typicality_normed{i})-0.08*deltay max(typicality_normed{i})+0.08*deltay])
    axis square
    ax = gca;
    ax.XTick = []; ax.YTick = [];
    hy(i) = ylabel(categories27{category27_subind((sortindtmp(i)))});
    hx(i) = xlabel(labels_short{best_match27(category27_subind(sortindtmp(i)))});
end

subtightplot(3,6,19,0.05)
ht(end+1) = text(0.9,0.05,'Spearman''s \rho','HorizontalAlignment','right');
% plot(0.5,0.5,'o','MarkerEdgeColor','none')
xlim([0 1]), ylim([0 1])
axis square
ax = gca;
ax.XTick = []; ax.YTick = [];
hy(end+1) = ylabel('Typicality scale');
hx(end+1) = xlabel('Dimension');

% set([ht hx hy],'FontName','Myriad Pro','FontSize',16,'Color',[0 0 0])
% set([hx hy],'HorizontalAlignment','center')

set(ht(p_typicality_s_adjusted(sortindtmp)<0.05),'FontWeight','bold')


% Bootstrap (uncorrected) confidence intervals
rng(1)
for i = 1:length(category27_subind)
    % using the unsorted one for both makes it easiest
    c1 = category27_typicality_rating_normed{category27_subind(i)};
    c2 = spose_embedding66(category27_ind{category27_subind(i)},best_match27(category27_subind(i)));
    nc = length(c1);
    rnd = randi(nc,nc,1000);
    for j = 1:1000
    r_typicality_s_boot(i,j) = corr(c1(rnd(:,j)), c2(rnd(:,j)) ,'type','spearman');
    end
end
r_typicality_s_ci95_lower = tanh(atanh(r_typicality_s) - 1.645*std(atanh(r_typicality_s_boot),[],2)); % reflects one-sided 95% CI (still unsorted)
r_typicality_s_ci95_upper = tanh(atanh(r_typicality_s) + 1.645*std(atanh(r_typicality_s_boot),[],2)); % reflects one-sided 95% CI


%% Get consistency of dimensions

dosave = 0;

load(fullfile(variable_dir,'sortind.mat')); % need this because original order is wrong
refdir = fullfile(base_dir,'models');
for i_model = 1:72
    if i_model == 64
        continue
    end
    fn = dir(fullfile(refdir,sprintf('embedding%02i_epoch0500.txt',i_model-1)));
    fn = fullfile(fn(end).folder,fn(end).name);
    tmp = load(fn);
    % remove empty dimensions
    tmp = tmp';
    reference_models{i_model,1} = tmp(:,any(tmp>0.1));
    n_dim_reference(i_model) = size(reference_models{i_model},2);
end
reference_models(64) = [];
n_dim_reference(64) = [];

% Correlate dimensions (this slightly overestimates the performance, given
% that each dimension can be picked several times, but there is no other
% way - otherwise some dimensions would go unmatched)
for i_model = 1:71
    reproducibility(:,i_model) = max(corr(spose_embedding66,reference_models{i_model}),[],2);
end

% test split-half prediction
for i_model = 1:71
    [~,maxind(:,i_model)] = max(corr(spose_embedding66(1:2:end,:),reference_models{i_model}(1:2:end,:)),[],2);
    [~,maxind2(:,i_model)] = max(corr(spose_embedding66(2:2:end,:),reference_models{i_model}(2:2:end,:)),[],2);
    c1 = corr(spose_embedding66(1:2:end,:),reference_models{i_model}(1:2:end,:));
    c2 = corr(spose_embedding66(2:2:end,:),reference_models{i_model}(2:2:end,:));
    for i = 1:66, tmp1(i,i_model) = c1(i,maxind2(i,i_model)); tmp2(i,i_model) = c2(i,maxind(i,i_model)); end
end

% fisher-z convert before averaging across models
mean_reproducibility = mean(atanh(reproducibility),2);
reproducibility_ci95 = 1.96*std(atanh(reproducibility),[],2)/sqrt(20);

% for plotting, the upper bound will be mean + 95% CI, then conversion back
% to correlation, same for lower bound
upper_bound = tanh(mean_reproducibility+reproducibility_ci95);
lower_bound = tanh(mean_reproducibility-reproducibility_ci95);
% now update mean reproducibility, as well
mean_reproducibility = tanh(mean_reproducibility);

hf = figure;
hf.Position(3:4) = [1024 768];
hold on
x = [1:66 66:-1:1];
y = [lower_bound' upper_bound(end:-1:1)'];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
plot(mean_reproducibility,'k','linewidth',1)
plot(reproducibility,'o','MarkerFaceColor',[0 0 0],'MarkerEdgeColor','none','MarkerSize',3)
ylim([0 1])
xlim([0 67])

% Test correlation between rank of reliability and dimension number
[~,reproducibility_ind] = sort(mean_reproducibility,'descend');
r_rank = corr((1:66)',reproducibility_ind);

% run 100000 permutations
rng(1)
[~,perm] = sort(rand(66,100000));
r_rank_perm = corr(perm,reproducibility_ind);
% is obviously never exceeded (smaller sign because it's a negative correlation)
p = mean([r_rank_perm;r_rank] >= r_rank);

% run 1000 bootstrap samples for confidence intervals
rng(2)
rnd = randi(66,66,1000);
for i = 1:1000
    r_rank_boot(:,i) = corr(rnd(:,i),reproducibility_ind(rnd(:,i)));
end
r_rank_ci95_lower = tanh(atanh(r_rank) - 1.96*std(atanh(r_rank_boot),[],2)); % reflects 95% CI
r_rank_ci95_upper = tanh(atanh(r_rank) + 1.96*std(atanh(r_rank_boot),[],2)); % reflects 95% CI