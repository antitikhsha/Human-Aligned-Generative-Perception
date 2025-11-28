warning('This code will not run before you haven''t downloaded the external datasets.')
disp('See content of "external_data" for details...')
pause(3)

dosave = 1;

base_dir = pwd;
addpath(genpath(fullfile(base_dir,'helper_functions')))
external_data_dir = fullfile(base_dir,'external_data');
data_dir = fullfile(base_dir,'data');

animal_names = {'bear','cat','deer','duck','parrot','seal','snake','tiger','turtle','whale'};
vehicle_names = {'airplane','bicycle','boat','car','helicopter','motorcycle','rocket','shuttle','submarine','truck'};
vehicle_names_adapted = {'airplane','bike','boat','car','helicopter','motorcycle','rocket','space shuttle','submarine','truck'};

% let's get Catalin Iordan's similarity matrices:
load(fullfile(external_data_dir,'iordan','animal-unconstrained-raw.mat'))
data_animal = double(data);
animal_sim_rating = squareformq(mean(data_animal));
load(fullfile(external_data_dir,'iordan','vehicle-unconstrained-raw.mat'))
data_vehicle = double(data);
vehicle_sim_rating = squareformq(mean(data_vehicle));

% as a baseline, also include sense vector
load(fullfile(data_dir,'sensevec_augmented_with_wordvec.mat'))

% let's get our similarity matrices for original and for the larger dataset
spose49 = load(fullfile(data_dir,'embedding49','spose_embedding_49d_sorted.txt'));
spose66 = load(fullfile(data_dir,'spose_embedding_66d_sorted.txt'));

load(fullfile(base_dir,'variables/words.mat'));

load(fullfile(data_dir,'typicality_data27.mat'), 'categories27')

[~,animal_ind] = intersect(words,animal_names);
[~,vehicle_ind] = intersect(words,vehicle_names_adapted);


sensevec_animal = sensevec_augmented(animal_ind,:);
sensevec_vehicle = sensevec_augmented(vehicle_ind,:);
% cosine similarity is normal
sensevec_animal_sim = 1-squareformq(pdist(sensevec_animal,'cos'));
sensevec_vehicle_sim = 1-squareformq(pdist(sensevec_vehicle,'cos'));

load(fullfile(data_dir,'embedding49','spose_similarity49.mat'));
spose_sim49 = spose_sim;
load(fullfile(data_dir,'spose_similarity.mat'));
spose_sim66 = spose_sim;

% for high level categories, get all animals and all vehicles
load(fullfile('data','category_mat_manual.mat'))
% 1 is animal, 26 is vehicle
spose_sim_animal49 = embedding2sim(spose49(category_mat_manual(:,1),:));
spose_sim_animal66 = embedding2sim(spose66(category_mat_manual(:,1),:));
spose_sim_vehicle49 = embedding2sim(spose49(category_mat_manual(:,26),:));
spose_sim_vehicle66 = embedding2sim(spose66(category_mat_manual(:,26),:));

% subindex animals and vehicles
animal_subind = find(category_mat_manual(:,1));
[~,animal_subind] = intersect(animal_subind,animal_ind);
vehicle_subind = find(category_mat_manual(:,26));
[~,vehicle_subind] = intersect(vehicle_subind,vehicle_ind);

% now extract subset that belongs to right category set

spose_sim_animal49_10 = spose_sim_animal49(animal_subind,animal_subind);
spose_sim_animal66_10 = spose_sim_animal66(animal_subind,animal_subind);

spose_sim_vehicle49_10 = spose_sim_vehicle49(vehicle_subind,vehicle_subind);
spose_sim_vehicle66_10 = spose_sim_vehicle66(vehicle_subind,vehicle_subind);



%% Now test similarity for other datasets by Peterson

animal_names_peterson = table2cell(readtable(fullfile(external_data_dir,'peterson','animal_labels.txt'),'ReadVariableNames',0));
% we had already adapted the animal names to get a stronger overlap with
% ours (e.g. an ocelot is categorized as a leopard, a mandrill as a monkey)
% but there are still some categories we should probably remove:
% bobcat, chimpanzee, lemur
animal_keepind = find(~ismember(animal_names_peterson,{'bobcat','chimpanzee','lemur'}));
load(fullfile(external_data_dir,'peterson','turkResults_CogSci2016.mat'),'simMatrix')
animal_sim_peterson = simMatrix(animal_keepind,animal_keepind);
animal_names_peterson = animal_names_peterson(animal_keepind);
% get index within 1854 words
for i = length(animal_names_peterson):-1:1
    animal_p_ind(i,1) = find(strcmp(words,animal_names_peterson{i}));
end
sensevec_animal_p = sensevec_augmented(animal_p_ind,:);
% cosine similarity is normal
sensevec_animal_p_sim = 1-squareformq(pdist(sensevec_animal_p,'cos'));

fruits = {'ackee','apple','apricot','atemoya','babaco','bacuri','banana','blackberry','blueberry','cantaloupe','carambola','cherry','coconut','cranberry','date','deadmansfinger','dragonfruit','durian','fig','grape','guava','kiwi','lemon','lime','lychee','mango','orange','papaya','passionfruit','peach','pear','persimmon','pineapple','pineberry','plum','pomegranate','pomelo','raspberry','strawberry','watermelon'};
% 24 is clearly enough
fruits_adapted = {'ackee','apple','apricot','atemoya','babaco','bacuri','banana','blackberry','blueberry','cantaloupe','carambola','cherry','coconut','cranberry','date','deadmansfinger','dragonfruit','durian','fig','grape','guava','kiwi','lemon','lime','lychee','mango','orange','papaya','passionfruit','peach','pear','persimmon','pineapple','pineberry','plum','pomegranate','pomelo','raspberry','strawberry','watermelon'};
load(fullfile(external_data_dir,'peterson/datasets/fruits/fruits_mturk_sim.mat'))
fruit_sim_peterson = simmat;
fruit_names_peterson = table2cell(readtable(fullfile(external_data_dir,'peterson','fruit_labels.txt'),'ReadVariableNames',0));

% fruit is category 11
fruit_keepind = ismember(fruit_names_peterson,words(category_mat_manual(:,11)));
fruit_sim_peterson = fruit_sim_peterson(fruit_keepind,fruit_keepind);
fruit_names_peterson = fruit_names_peterson(fruit_keepind);
% get index within 1854 words
for i = length(fruit_names_peterson):-1:1
    fruit_p_ind(i,1) = find(strcmp(words,fruit_names_peterson{i}));
end
sensevec_fruit_p = sensevec_augmented(fruit_p_ind,:);
% cosine similarity is normal
sensevec_fruit_p_sim = 1-squareformq(pdist(sensevec_fruit_p,'cos'));

furniture =     {'bar-stool','bed','bench','bookshelf','chairs','closet','coffee-table','cupboard','desk','drawer','end-table','lawn-chair','office-chair','ottoman','recliner','sofa','tables','vanity'};
% 12 might not be enough
furniture_adapted = {'stool','bed','bench','bookshelf','chair','closet','coffee table','cupboard','desk','drawer','end-table','lawn-chair','office-chair','ottoman','recliner','couch','table','vanity'};
load(fullfile(external_data_dir,'peterson/datasets/furniture/simMatrix_mturkfurniture.mat'))
furniture_sim_peterson = simMatrix_mturkfurniture;
furniture_names_peterson = table2cell(readtable(fullfile(external_data_dir,'peterson','furniture_labels.txt'),'ReadVariableNames',0));
% furniture is category 12
furniture_keepind = ismember(furniture_names_peterson,words(category_mat_manual(:,12)));
furniture_sim_peterson = furniture_sim_peterson(furniture_keepind,furniture_keepind);
furniture_names_peterson = furniture_names_peterson(furniture_keepind);
% get index within 1854 words
for i = length(furniture_names_peterson):-1:1
    furniture_p_ind(i,1) = find(strcmp(words,furniture_names_peterson{i}));
end
sensevec_furniture_p = sensevec_augmented(furniture_p_ind,:);
% cosine similarity is normal
sensevec_furniture_p_sim = 1-squareformq(pdist(sensevec_furniture_p,'cos'));

vegetables = {'amaranth','artichoke','arugula','asparagus','beetroot','bellpepper','broccoli','brusselssprouts','cabbage','carrot','catsear','cauliflower','celery','celtuce','chillipeppers','chives','corn','cucumber','daikon','eggplant','ginger','leeks','lettuce','luffa','pea','potatoes','pumpkin','radicchio','redonions','redskinpotato','redswanbean','rhubarb','rutabaga','salsify','seaweed','tatsoi','thaieggplant','tomato','wintersquash','yellowonion'};
% intersection of 23, which is enough
vegetables_adapted = {'amaranth','artichoke','arugula','asparagus','beet','bell pepper','broccoli','brussels sprouts','cabbage','carrot','catsear','cauliflower','celery','celtuce','chillipeppers','chive','corn','cucumber','daikon','eggplant','ginger','leek','lettuce','luffa','pea','potato','pumpkin','radicchio','redonions','redskinpotato','redswanbean','rhubarb','rutabaga','salsify','seaweed','tatsoi','thaieggplant','tomato','wintersquash','onion'};
load(fullfile(external_data_dir,'peterson/datasets/vegetables/simMatrix_mturkveggies.mat'))
vegetable_sim_peterson = simMatrix_mturkveggies;
vegetable_names_peterson = table2cell(readtable(fullfile(external_data_dir,'peterson','vegetable_labels.txt'),'ReadVariableNames',0));
% vegetable is category 25
vegetable_keepind = ismember(vegetable_names_peterson,words(category_mat_manual(:,25)));
vegetable_sim_peterson = vegetable_sim_peterson(vegetable_keepind,vegetable_keepind);
vegetable_names_peterson = vegetable_names_peterson(vegetable_keepind);
% get index within 1854 words
for i = length(vegetable_names_peterson):-1:1
    vegetable_p_ind(i,1) = find(strcmp(words,vegetable_names_peterson{i}));
end
sensevec_vegetable_p = sensevec_augmented(vegetable_p_ind,:);
% cosine similarity is normal
sensevec_vegetable_p_sim = 1-squareformq(pdist(sensevec_vegetable_p,'cos'));

vehicles =         {'airplane','bike','blimp','boat','bus','car','cart','elevator','horse','motorcycle','raft','skates','sled','tank','tractor','train','trolleycar','truck','wheelbarrow','wheelchair'};
vehicles_adapted = {'airplane','bike','blimp','boat','bus','car','cart','elevator','horse','motorcycle','raft','rollerskate','sled','tank','tractor','train','trolleycar','truck','wheelbarrow','wheelchair'};
% keeping only the examples overlapping with THINGS categories, there are only 13 remaining examples
load(fullfile(external_data_dir,'peterson/datasets/automobiles/simMatrix_mturkvehicles.mat'))
vehicle_sim_peterson = simMatrix_mturkvehicles;
vehicle_names_peterson = table2cell(readtable(fullfile(external_data_dir,'peterson','vehicle_labels.txt'),'ReadVariableNames',0));
% vehicles is category 26
vehicle_keepind = ismember(vehicle_names_peterson,words(category_mat_manual(:,26)));
vehicle_sim_peterson = vehicle_sim_peterson(vehicle_keepind,vehicle_keepind);
vehicle_names_peterson = vehicle_names_peterson(vehicle_keepind);
% get index within 1854 words
for i = length(vehicle_names_peterson):-1:1
    vehicle_p_ind(i,1) = find(strcmp(words,vehicle_names_peterson{i}));
end
sensevec_vehicle_p = sensevec_augmented(vehicle_p_ind,:);
% cosine similarity is normal
sensevec_vehicle_p_sim = 1-squareformq(pdist(sensevec_vehicle_p,'cos'));

%% 

% 1 is animal, fruit is 11, furniture is 12, vegetable is 25, vehicle is 26
spose_sim_animal49 = embedding2sim(spose49(category_mat_manual(:,1),:));
spose_sim_animal66 = embedding2sim(spose66(category_mat_manual(:,1),:));

spose_sim_fruit49 = embedding2sim(spose49(category_mat_manual(:,11),:));
spose_sim_fruit66 = embedding2sim(spose66(category_mat_manual(:,11),:));

spose_sim_furniture49 = embedding2sim(spose49(category_mat_manual(:,12),:));
spose_sim_furniture66 = embedding2sim(spose66(category_mat_manual(:,12),:));

spose_sim_vegetable49 = embedding2sim(spose49(category_mat_manual(:,25),:));
spose_sim_vegetable66 = embedding2sim(spose66(category_mat_manual(:,25),:));

spose_sim_vehicle49 = embedding2sim(spose49(category_mat_manual(:,26),:));
spose_sim_vehicle66 = embedding2sim(spose66(category_mat_manual(:,26),:));

% subindex animals, fruit, furniture, vegetables, and vehicles
tmp = find(category_mat_manual(:,1));
for i = length(animal_p_ind):-1:1
    animal_p_subind(i) = find(tmp==animal_p_ind(i));
end
tmp = find(category_mat_manual(:,11));
for i = length(fruit_p_ind):-1:1
    fruit_p_subind(i) = find(tmp==fruit_p_ind(i));
end
tmp = find(category_mat_manual(:,12));
for i = length(furniture_p_ind):-1:1
    furniture_p_subind(i) = find(tmp==furniture_p_ind(i));
end
tmp = find(category_mat_manual(:,25));
for i = length(vegetable_p_ind):-1:1
    vegetable_p_subind(i) = find(tmp==vegetable_p_ind(i));
end
tmp = find(category_mat_manual(:,26));
for i = length(vehicle_p_ind):-1:1
    vehicle_p_subind(i) = find(tmp==vehicle_p_ind(i));
end


% now extract subset that belongs to right category set
spose_sim_animal_p49 = spose_sim_animal49(animal_p_subind,animal_p_subind);
spose_sim_animal_p66 = spose_sim_animal66(animal_p_subind,animal_p_subind);

spose_sim_fruit_p49 = spose_sim_fruit49(fruit_p_subind,fruit_p_subind);
spose_sim_fruit_p66 = spose_sim_fruit66(fruit_p_subind,fruit_p_subind);

spose_sim_furniture_p49 = spose_sim_furniture49(furniture_p_subind,furniture_p_subind);
spose_sim_furniture_p66 = spose_sim_furniture66(furniture_p_subind,furniture_p_subind);

spose_sim_vegetable_p49 = spose_sim_vegetable49(vegetable_p_subind,vegetable_p_subind);
spose_sim_vegetable_p66 = spose_sim_vegetable66(vegetable_p_subind,vegetable_p_subind);

spose_sim_vehicle_p49 = spose_sim_vehicle49(vehicle_p_subind,vehicle_p_subind);
spose_sim_vehicle_p66 = spose_sim_vehicle66(vehicle_p_subind,vehicle_p_subind);

%% Now get Jason's data in order

food_names_avery = table2cell(readtable(fullfile(external_data_dir,'avery','foodnames.txt'),'ReadVariableNames',0,'Delimiter','\n'));

% now load in triplets and assign to matrix

tab1 = readtable(fullfile(external_data_dir,'avery','food_triplet_data_set1.csv'));
tab2 = readtable(fullfile(external_data_dir,'avery','food_triplet_data_set2.csv'));
tabx = table2cell([tab1;tab2]);
taby = zeros(size(tabx));


for i = 1:length(food_names_avery)
    ind = strcmp(tabx,food_names_avery{i});
    taby(ind) = i;
end
taby(:,[1 6]) = [];

% now replace food names
food_names_avery = regexprep(food_names_avery,'\_[0-9]','');
food_names_avery = strrep(food_names_avery,'doughnut','donut');
food_names_avery = strrep(food_names_avery,'_',' ');
food_names_avery = strrep(food_names_avery,'hot dog','hotdog');
food_names_avery = strrep(food_names_avery,'chocolate candy','chocolate');
% food is category 10
food_keepind = find(ismember(food_names_avery,words(category_mat_manual(:,10))));
food_names_avery = food_names_avery(food_keepind);

    
choicemat = zeros(36,36);
cntmat = eye(36,36);   
for i = 1:length(taby)
    currpair = setdiff(taby(i,1:3),taby(i,4));
    choicemat(currpair(1),currpair(2)) = choicemat(currpair(1),currpair(2))+1;
    choicemat(currpair(2),currpair(1)) = choicemat(currpair(2),currpair(1))+1;
    cntmat(taby(i,1),taby(i,2)) = cntmat(taby(i,1),taby(i,2))+1;
    cntmat(taby(i,2),taby(i,1)) = cntmat(taby(i,2),taby(i,1))+1;
    cntmat(taby(i,1),taby(i,3)) = cntmat(taby(i,1),taby(i,3))+1;
    cntmat(taby(i,3),taby(i,1)) = cntmat(taby(i,3),taby(i,1))+1;
    cntmat(taby(i,2),taby(i,3)) = cntmat(taby(i,2),taby(i,3))+1;
    cntmat(taby(i,3),taby(i,2)) = cntmat(taby(i,3),taby(i,2))+1;
end
food_sim_avery = choicemat./cntmat;
food_sim_avery = food_sim_avery(food_keepind,food_keepind);
    
% get index within 1854 words
for i = length(food_names_avery):-1:1
    food_a_ind(i,1) = find(strcmp(words,food_names_avery{i}));
end
sensevec_food_a = sensevec_augmented(food_a_ind,:);
% cosine similarity is normal
sensevec_food_a_sim = 1-squareformq(pdist(sensevec_food_a,'cos'));    

% subindex food
tmp = find(category_mat_manual(:,10));
for i = length(food_a_ind):-1:1
    food_a_subind(i) = find(tmp==food_a_ind(i));
end

spose_sim_food49 = embedding2sim(spose49(category_mat_manual(:,10),:));
spose_sim_food66 = embedding2sim(spose66(category_mat_manual(:,10),:));

spose_sim_food_a49 = spose_sim_food49(food_a_subind,food_a_subind);
spose_sim_food_a66 = spose_sim_food66(food_a_subind,food_a_subind);

%%

% finally, look at prediction
clear cc

% look at prediction

clear cc

% Animal Iordan 
cc(1,1) = corr(squareformq(sensevec_animal_sim),squareformq(animal_sim_rating));
cc(1,2) = corr(squareformq(spose_sim_animal49_10),squareformq(animal_sim_rating));
cc(1,3) = corr(squareformq(spose_sim_animal66_10),squareformq(animal_sim_rating));

% Animal Peterson
cc(2,1) = corr(squareformq(sensevec_animal_p_sim),squareformq(animal_sim_peterson));
cc(2,2) = corr(squareformq(spose_sim_animal_p49),squareformq(animal_sim_peterson));
cc(2,3) = corr(squareformq(spose_sim_animal_p66),squareformq(animal_sim_peterson));

% Food Avery
cc(3,1) = corr(squareformq(sensevec_food_a_sim),squareformq(food_sim_avery));
cc(3,2) = corr(squareformq(spose_sim_food_a49),squareformq(food_sim_avery));
cc(3,3) = corr(squareformq(spose_sim_food_a66),squareformq(food_sim_avery));

% Fruit Peterson
cc(4,1) = corr(squareformq(sensevec_fruit_p_sim),squareformq(fruit_sim_peterson));
cc(4,2) = corr(squareformq(spose_sim_fruit_p49),squareformq(fruit_sim_peterson));
cc(4,3) = corr(squareformq(spose_sim_fruit_p66),squareformq(fruit_sim_peterson));

% % Furniture Peterson
cc(5,1) = corr(squareformq(sensevec_furniture_p_sim),squareformq(furniture_sim_peterson));
cc(5,2) = corr(squareformq(spose_sim_furniture_p49),squareformq(furniture_sim_peterson));
cc(5,3) = corr(squareformq(spose_sim_furniture_p66),squareformq(furniture_sim_peterson));

% Vegetable Peterson
cc(6,1) = corr(squareformq(sensevec_vegetable_p_sim),squareformq(vegetable_sim_peterson));
cc(6,2) = corr(squareformq(spose_sim_vegetable_p49),squareformq(vegetable_sim_peterson));
cc(6,3) = corr(squareformq(spose_sim_vegetable_p66),squareformq(vegetable_sim_peterson));

% Vehicle Iordan
cc(7,1) = corr(squareformq(sensevec_vehicle_sim),squareformq(vehicle_sim_rating));
cc(7,2) = corr(squareformq(spose_sim_vehicle49_10),squareformq(vehicle_sim_rating));
cc(7,3) = corr(squareformq(spose_sim_vehicle66_10),squareformq(vehicle_sim_rating));

% % Vehicle Peterson
cc(8,1) = corr(squareformq(sensevec_vehicle_p_sim),squareformq(vehicle_sim_peterson));
cc(8,2) = corr(squareformq(spose_sim_vehicle_p49),squareformq(vehicle_sim_peterson));
cc(8,3) = corr(squareformq(spose_sim_vehicle_p66),squareformq(vehicle_sim_peterson));



%% get confidence interval by bootstrapping
rng(1)
% commented out all the unnecessary ones for speed
for ii = 100000:-1:1 % 100k iterations for better estimation of p-value
    
    if ~mod(ii,10000), disp(ii), end
    
    n = nchoosek(length(sensevec_animal_sim),2);
    ind = randi(n,n,1);
    
    % look at prediction
%     cc_boot(1,1,ii) = corr(squareformq_sub(sensevec_animal_sim,ind),squareformq_sub(animal_sim_rating,ind));
    cc_boot(1,2,ii) = corr(squareformq_sub(spose_sim_animal49_10,ind),squareformq_sub(animal_sim_rating,ind));
    cc_boot(1,3,ii) = corr(squareformq_sub(spose_sim_animal66_10,ind),squareformq_sub(animal_sim_rating,ind));
    
    n = nchoosek(length(sensevec_animal_p_sim),2);
    ind = randi(n,n,1);
    
%     cc_boot(2,1,ii) = corr(squareformq_sub(sensevec_animal_p_sim,ind),squareformq_sub(animal_sim_peterson,ind));
    cc_boot(2,2,ii) = corr(squareformq_sub(spose_sim_animal_p49,ind),squareformq_sub(animal_sim_peterson,ind));
    cc_boot(2,3,ii) = corr(squareformq_sub(spose_sim_animal_p66,ind),squareformq_sub(animal_sim_peterson,ind));
    
    n = nchoosek(length(sensevec_food_a_sim),2);
    ind = randi(n,n,1);
    
%     cc_boot(3,1,ii) = corr(squareformq_sub(sensevec_food_a_sim,ind),squareformq_sub(food_sim_avery,ind));
    cc_boot(3,2,ii) = corr(squareformq_sub(spose_sim_food_a49,ind),squareformq_sub(food_sim_avery,ind));
    cc_boot(3,3,ii) = corr(squareformq_sub(spose_sim_food_a66,ind),squareformq_sub(food_sim_avery,ind));
    
    n = nchoosek(length(sensevec_fruit_p_sim),2);
    ind = randi(n,n,1);
    
%     cc_boot(4,1,ii) = corr(squareformq_sub(sensevec_fruit_p_sim,ind),squareformq_sub(fruit_sim_peterson,ind));
    cc_boot(4,2,ii) = corr(squareformq_sub(spose_sim_fruit_p49,ind),squareformq_sub(fruit_sim_peterson,ind));
    cc_boot(4,3,ii) = corr(squareformq_sub(spose_sim_fruit_p66,ind),squareformq_sub(fruit_sim_peterson,ind));
    
    n = nchoosek(length(sensevec_furniture_p_sim),2);
    ind = randi(n,n,1);
    
%     cc_boot(5,1,ii) = corr(squareformq_sub(sensevec_furniture_p_sim,ind),squareformq_sub(furniture_sim_peterson,ind));
    cc_boot(5,2,ii) = corr(squareformq_sub(spose_sim_furniture_p49,ind),squareformq_sub(furniture_sim_peterson,ind));
    cc_boot(5,3,ii) = corr(squareformq_sub(spose_sim_furniture_p66,ind),squareformq_sub(furniture_sim_peterson,ind));
    
    n = nchoosek(length(sensevec_vegetable_p_sim),2);
    ind = randi(n,n,1);
    
%     cc_boot(6,1,ii) = corr(squareformq_sub(sensevec_vegetable_p_sim,ind),squareformq_sub(vegetable_sim_peterson,ind));
    cc_boot(6,2,ii) = corr(squareformq_sub(spose_sim_vegetable_p49,ind),squareformq_sub(vegetable_sim_peterson,ind));
    cc_boot(6,3,ii) = corr(squareformq_sub(spose_sim_vegetable_p66,ind),squareformq_sub(vegetable_sim_peterson,ind));
    
    n = nchoosek(length(sensevec_vehicle_sim),2);
    ind = randi(n,n,1);
    
%     cc_boot(7,1,ii) = corr(squareformq_sub(sensevec_vehicle_sim,ind),squareformq_sub(vehicle_sim_rating,ind));
    cc_boot(7,2,ii) = corr(squareformq_sub(spose_sim_vehicle49_10,ind),squareformq_sub(vehicle_sim_rating,ind));
    cc_boot(7,3,ii) = corr(squareformq_sub(spose_sim_vehicle66_10,ind),squareformq_sub(vehicle_sim_rating,ind));
    
    n = nchoosek(length(sensevec_vehicle_p_sim),2);
    ind = randi(n,n,1);
    
%     cc_boot(8,1,ii) = corr(squareformq_sub(sensevec_vehicle_p_sim,ind),squareformq_sub(vehicle_sim_peterson,ind));
    cc_boot(8,2,ii) = corr(squareformq_sub(spose_sim_vehicle_p49,ind),squareformq_sub(vehicle_sim_peterson,ind));
    cc_boot(8,3,ii) = corr(squareformq_sub(spose_sim_vehicle_p66,ind),squareformq_sub(vehicle_sim_peterson,ind));
    
end

% Sample one bootstrap sample at each level
% Then compute fraction of the number of tests with an improvement and see
% whether across all 8 experiments there is an improvement or not
p_val = sum(squeeze(mean((cc_boot(:,3,:)-cc_boot(:,2,:))>0))<=0.5)/100000;
fprintf('p-value: %.6f\n',p_val)

%% Make figure

cc2 = squeeze(std(cc_boot(:,2,:),[],3));
cc3 = squeeze(std(cc_boot(:,3,:),[],3));
cc2 = repmat(cc2,1,2);
cc3 = repmat(cc3,1,2);
% Shape: source of dataset
% Color: category
figure,
plot([0 1],[0 1],'k:')
hold on
% animal, green
h(1) = errorbar(cc(1,2),cc(1,3),cc2(1,1),cc2(1,2),cc3(1,1),cc3(1,2),'v','MarkerFaceColor',[0 0.8 0],'MarkerEdgeColor','none','MarkerSize',10);
h(1).Color = [0.7 0.7 0.7];
% animal, green
h(2) = errorbar(cc(2,2),cc(2,3),cc2(2,1),cc2(2,2),cc3(2,1),cc3(2,2),'o','MarkerFaceColor',[0 0.8 0],'MarkerEdgeColor','none','MarkerSize',10);
h(2).Color = [0.7 0.7 0.7];
% food, orange
h(3) = errorbar(cc(3,2),cc(3,3),cc2(3,1),cc2(3,2),cc3(3,1),cc3(3,2),'s','MarkerFaceColor',[1 0.5 0],'MarkerEdgeColor','none','MarkerSize',10);
h(3).Color = [0.7 0.7 0.7];
% fruit, red
h(4) = errorbar(cc(4,2),cc(4,3),cc2(4,1),cc2(4,2),cc3(4,1),cc3(4,2),'o','MarkerFaceColor',[0.8 0 0],'MarkerEdgeColor','none','MarkerSize',10);
h(4).Color = [0.7 0.7 0.7];
% furniture, brown
h(5) = errorbar(cc(5,2),cc(5,3),cc2(5,1),cc2(5,2),cc3(5,1),cc3(5,2),'o','MarkerFaceColor',[0.8 0.8 0],'MarkerEdgeColor','none','MarkerSize',10);
h(5).Color = [0.7 0.7 0.7];
% vegetable, dark green
h(6) = errorbar(cc(6,2),cc(6,3),cc2(6,1),cc2(6,2),cc3(6,1),cc3(6,2),'o','MarkerFaceColor',[0 0.4 0],'MarkerEdgeColor','none','MarkerSize',10);
h(6).Color = [0.7 0.7 0.7];
% vehicle, blue
h(7) = errorbar(cc(7,2),cc(7,3),cc2(7,1),cc2(7,2),cc3(7,1),cc3(7,2),'v','MarkerFaceColor',[0 0 1],'MarkerEdgeColor','none','MarkerSize',10);
h(7).Color = [0.7 0.7 0.7];
% vehicle, blue
h(8) = errorbar(cc(8,2),cc(8,3),cc2(8,1),cc2(8,2),cc3(8,1),cc3(8,2),'o','MarkerFaceColor',[0 0 1],'MarkerEdgeColor','none','MarkerSize',10);
h(8).Color = [0.7 0.7 0.7];

xlim([0.35 1]),ylim([0.35 1])
axis square

title('Prediction of within-category similarity in external datasets')
xlabel('Previous dataset (1.46 million triplets)')
ylabel('Current dataset (4.70 million triplets)')


set(gcf,'Renderer','painters')

if dosave
    print(gcf,'-dpdf','temp1.pdf','-bestfit')
end

fprintf('Mean improvement in correlation when using 66d vs. 49d: %.3f\x00B1%.3f\n',mean(cc(:,3)-cc(:,2)),std(squeeze(mean(cc_boot(:,3,:)-cc_boot(:,2,:)))))