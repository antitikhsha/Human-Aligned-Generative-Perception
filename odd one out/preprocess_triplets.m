% This code shows all the preprocessing that took place for converting the
% raw data to the processed data (list of all triplets as a csv) and then 
% converting the preprocessed data to training, validation and test sets 
% (the latter of which are used for noise ceilings).
% Since participant IDs (worker IDs) act as pseudonyms, they were recoded
% in this process using a key that was later deleted, rendering the
% identity of workers anonymous.

clear variables

% make key for recoding workers

tmp = rng('shuffle');
rng(tmp)
save tmp.mat tmp

key = randi(36,1,21);

ubercounter = 0; % counts HITs across all batches

%% Now save original 1.46 million triplets

base_dir = '/Users/hebart/Documents/projects/hebart/behavsim/experiment1854/results/results_orig';
fnames_orig = dir(fullfile(base_dir,'*.csv'));
fnames_orig = {fnames_orig.name}';
fnames_orig = strcat(repmat({[base_dir '/']},length(fnames_orig),1),fnames_orig);

disp('Original triplets:')

for i_name = 1:length(fnames_orig)
    
%     disp(i_name)
    
    filename = fnames_orig{i_name};

%% Initialize variables.
delimiter = ',';
startRow = 2;
endRow = inf;

%% Open the text file.
fileID = fopen(filename,'r');

%% figure out the number of columns
header = fgetl(fileID);
header = strrep(header,'.','_'); % replace . with _
header = strrep(header,'"',''); % remove ""
variable_names = strsplit(header,',');

frewind(fileID);

%% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = [repmat('%q',1,length(variable_names)) '%[^\n\r]'];

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

% batchstr = strsplit(dataArray{9}{1},';');
% batch_nr_orig(i,1) = str2double(batchstr{1}(9:end));

if length(dataArray)>length(variable_names)
    if ~all(cellfun(@isempty,dataArray{end})), error('mismatch between variable names and dataArray size'), end
    
    dataArray = dataArray(1:length(variable_names)); % remove empty column at back
    
end

% variables that we leave out
left_out = {'Approve','Reject','Answer_doNotRedirect'};

[~,rmind] = intersect(variable_names,left_out);

variable_names(rmind) = [];
dataArray(rmind) = [];

if length(dataArray) ~= length(variable_names)
    error('mismatch between variable names and dataArray size')
end

answer_field = find(~cellfun(@isempty,strfind(variable_names,'Answer')));

clear str
str(length(dataArray{1}),1).tmp = 0; % trick to initialize
for i = 1:length(variable_names)
    if any(answer_field==i), continue, end
    [str.(variable_names{i})] = deal(dataArray{i}{:});
end
str = rmfield(str,'tmp');

% gender, hispanic, race
ghr = {'gender','hispanic','race'};
for i = answer_field
    
    currind = ~cellfun(@isempty,regexp(variable_names{i},ghr));
    if any(currind)
        [str.(ghr{currind})] = deal(dataArray{i}{:});
    end
end

tmp = num2cell(str2double({str.WorkTimeInSeconds}'),2);
[str.duration_in_seconds] = deal(tmp{:});

currind = strcmp(variable_names,'Answer_expDuration');
cf = cellfun(@str2double,dataArray{currind})/1000;
cf = num2cell(cf,2);
[str.exp_duration] = deal(cf{:});

in_names = {'Answer_RT','Answer_question','Answer_imLink1','Answer_imLink2','Answer_imLink3'};
out_names = {'RT','choice','imlink1','imlink2','imlink3'};

% format choices
currind = strcmp(variable_names,'Answer_question');
dataArray(currind) = cellfun(@(x) strrep(x,'link',''),dataArray(currind),'uniformoutput',0);

for i = 1:length(in_names)
currind = strcmp(variable_names,in_names{i});
cf = cellfun(@(x) sscanf(x,'%f')',dataArray{currind},'uniformoutput',0);
% cf = num2cell(cf,2);
[str.(out_names{i})] = deal(cf{:});
end

[str.batch_id] = deal(i_name);

STR{i_name,1} = str;

end

str = vertcat(STR{:});

n = length(STR);

clear STR

%% check for exact duplicates and remove


hit_id = {str.HITId}';
batch_id = {str.batch_id}';
WorkerId = {str.WorkerId}';

[~,ind] = unique(hit_id);

m = ones(length(hit_id),1);
m(ind) = 0;

m_ind = find(m);

% now let's see if they are just the same or exact duplicates using hit_id AND WorkerId:

exact_duplicate_ind = [];
for i = 1:length(m_ind)
    duplicate_ind = find(strcmp(hit_id,hit_id(m_ind(i))));
    id = WorkerId(duplicate_ind);
    if strcmp(id(1),id(2))
        exact_duplicate_ind(end+1) = duplicate_ind(2);
    end
end
    
str(exact_duplicate_ind) = [];
fprintf('Number of HITs/trials removed due to duplicates: %i / %i\n',length(exact_duplicate_ind),length(exact_duplicate_ind)*20)



%% check for cheaters

[rmind,n_rm,n_unique] = check_for_cheaters(str);

%% check performance on original model of removed triplets
% fprintf('Performance of removed triplets: %.2f\n',check_performance_orig(str(rmind)))


%% now remove those from str

str(rmind) = [];
% fprintf('Performance of remaining triplets: %.2f\n',check_performance_orig(str))
fprintf('Number of HITs/trials removed due to speed or repetitiveness: %i / %i\n',length(rmind),length(rmind)*20)
fprintf('Number of workers removed from this dataset: %i / %i\n',n_rm,n_unique)

% now check for HITs that we remove
rmindnan = false(length(str),1);
for i = 1:length(str)
    if any(isnan(str(i).RT)) || isempty(str(i).RT)
        rmindnan(i) = true;
    end
end
str(rmindnan) = [];
fprintf('Number of HITs/trials removed due to being empty: %i / %i\n',sum(rmindnan),sum(rmindnan)*20)
fprintf('Total number of HITs/trials included: %i / %i\n',length(str),length(str)*20)

%% Let's write main results out

% we want to keep triplet, choice, RT, noise ceiling, ID, HIT_nr, trial_nr, age, gender, date, time 

h = fopen('triplets_large_temp_part1.csv','w');
fprintf(h,'image1\timage2\timage3\tchoice\tRT\tnoise_ceiling\tsubject_id\tHIT_nr\ttrial_nr\tage\tgender\tdate\ttime\n');

% recode worker ids to subject ids
subject_ids = recode_workerids({str.WorkerId},key);

for i = 1:length(str)
    if any(isnan(str(i).choice))
        keyboard
    end
    
    subject_id = subject_ids{i};
    
    imlink = [str(i).imlink1' str(i).imlink2' str(i).imlink3'];
    
    ubercounter = ubercounter + 1;
    
    curr_datestr = str(i).SubmitTime;
    curr_datestr = strrep(curr_datestr,'PST ','');
    curr_datestr = strrep(curr_datestr,'PDT ','');
    curr_datestr = [curr_datestr(end-3:end) ' ' curr_datestr(5:end-5)];
    curr_date = strrep(curr_datestr(1:end-9),' ','-');
    % go through months
    mon = {'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'};
    monnr = 1:12;
    for j = 1:12
        curr_date = strrep(lower(curr_date),lower(mon{j}),sprintf('%02i',monnr(j)));
    end
    curr_time = curr_datestr(end-7:end);
    
    nc = zeros(20,1);
    
    for j = 1:length(imlink)
        fprintf(h,'%i\t%i\t%i\t%i\t%i\t%i\t%s\t%i\t%i\t%i\t%s\t%s\t%s\n',...
            imlink(j,1),imlink(j,2),imlink(j,3),str(i).choice(j),str(i).RT(j),nc(j),subject_id,ubercounter,j,NaN,str(i).gender,curr_date,curr_time);
    end
    
end

fclose(h);

%% Next, save triplets for second test set (originally called noise ceiling)

disp('Original noise ceiling:')

save ubercounter.mat ubercounter

clear variables

load ubercounter.mat

load('tmp.mat')
rng(tmp)
key = randi(36,1,21);

res_path = '/Users/hebart/Documents/projects/hebart/behavsim/experiment1854/results/results_batch57a.csv';

h = fopen(res_path);

all = {};
while 1
    l = fgetl(h);
    if l == -1
        break
    end
    all{end+1} = l;
end

fclose(h);

for i = 1:length(all)
    all{i} = strrep(all{i},'""','" "'); % make empty cells not empty -> strsplit messes this up elsewise
    all{i} = strrep(all{i},'","','&&&'); % use &&& as unique identifier
    
    
    all{i}(1) = []; % will remove "
    all{i}(end) = [];
end

all{1} = strrep(all{1},'.','_'); % replace . for field names
names = strsplit(all{1},'&&&');
names(end-1:end) = []; % remove Approve and Reject fields

all(1) = []; % remove headers

ALL{1} = all;
NAMES{1} = names;

res_path = '/Users/hebart/Documents/projects/hebart/behavsim/experiment1854/results/results_batch57b.csv';

h = fopen(res_path);

all = {};
while 1
    l = fgetl(h);
    if l == -1
        break
    end
    all{end+1} = l;
end

fclose(h);

for i = 1:length(all)
    all{i} = strrep(all{i},'""','" "'); % make empty cells not empty -> strsplit messes this up elsewise
    all{i} = strrep(all{i},'","','&&&'); % use &&& as unique identifier
    
    
    all{i}(1) = []; % will remove "
    all{i}(end) = [];
end

all{1} = strrep(all{1},'.','_'); % replace . for field names
names = strsplit(all{1},'&&&');
names(end-1:end) = []; % remove Approve and Reject fields

all(1) = []; % remove headers

ALL{2} = all;
NAMES{2} = names;

%%

clear STR

for ii = 1:length(ALL)
    
    clear str
    
    all = ALL{ii};
    
    for i = length(all):-1:1
        
        curr_res = strsplit(all{i},'&&&');
        
        names = NAMES{ii};
        
        for j = 1:length(names)
            if any(strfind(names{j},'Answer')) % expand those values
                str(i).(names{j}) = strsplit(curr_res{j},' ');
                % for "question", we want to split the links first
            elseif any(strfind(names{j},'WorkerId'))
                str(i).(names{j}) = curr_res{j};
            elseif any(strfind(names{j},'AcceptTime'))
                str(i).(names{j}) = curr_res{j};
            elseif any(strfind(names{j},'SubmitTime'))
                str(i).(names{j}) = curr_res{j};
            elseif any(strfind(names{j},'HITId'))
                str(i).(names{j}) = curr_res{j};
            else
                
                str(i).Mturk.(names{j}) = curr_res{j};
                
            end
        end
        
        dates = strsplit(str(i).AcceptTime);
        startdate = datevec(sprintf('%02i-%s-%s %s',str2double(dates{3}),dates{2},dates{6},dates{4}));
        dates = strsplit(str(i).SubmitTime);
        enddate = datevec(sprintf('%02i-%s-%s %s',str2double(dates{3}),dates{2},dates{6},dates{4}));
        
        str(i).duration_in_seconds = etime(enddate,startdate);
        
    end
    
    % now convert relevant fields
    for i = 1:length(str)
        str(i).exp_duration = str2double(str(i).Answer_expDuration)/1000;
        str(i).RT = str2double(str(i).Answer_RT);
        str(i).choice = str2double(strrep(str(i).Answer_question,'link',''));
        str(i).imlink1 = str2double(str(i).Answer_imLink1);
        str(i).imlink2 = str2double(str(i).Answer_imLink2);
        str(i).imlink3 = str2double(str(i).Answer_imLink3);
    end
    
    
    % remove fields
    str = rmfield(str,{'Answer_expDuration','Answer_RT','Answer_imLink1','Answer_imLink2','Answer_imLink3','Answer_question'});
    try
        str = rmfield(str,{'Answer_doNotRedirect'});
    end
    
    STR{ii,1} = str;
    
end

str = horzcat(STR{:});

%% check for cheaters


[rmind,n_rm,n_unique] = check_for_cheaters(str);

%% check performance on original model of removed triplets
% fprintf('Performance of removed triplets: %.2f\n',check_performance_orig(str(rmind)))


%% now remove those from str

str(rmind) = [];
% fprintf('Performance of remaining triplets: %.2f\n',check_performance_orig(str))
fprintf('Number of HITs/trials removed due to speed or repetitiveness: %i / %i\n',length(rmind),length(rmind)*20)
fprintf('Number of workers removed from this dataset: %i / %i\n',n_rm,n_unique)

%%

h = fopen('triplets_large_temp_part2.csv','w');
fprintf(h,'image1\timage2\timage3\tchoice\tRT\tnoise_ceiling\tsubject_id\tHIT_nr\ttrial_nr\tage\tgender\tdate\ttime\n');

% recode worker ids to subject ids
subject_ids = recode_workerids({str.WorkerId},key);

for i = 1:length(str)
    if any(isnan(str(i).choice))
        keyboard
    end
    
    subject_id = subject_ids{i};
    
    imlink = [str(i).imlink1' str(i).imlink2' str(i).imlink3'];
    
    ubercounter = ubercounter + 1;
    
    curr_datestr = str(i).SubmitTime;
    curr_datestr = strrep(curr_datestr,'PST ','');
    curr_datestr = strrep(curr_datestr,'PDT ','');
    curr_datestr = [curr_datestr(end-3:end) ' ' curr_datestr(5:end-5)];
    curr_date = strrep(curr_datestr(1:end-9),' ','-');
    % go through months
    mon = {'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'};
    monnr = 1:12;
    for j = 1:12
        curr_date = strrep(lower(curr_date),lower(mon{j}),sprintf('%02i',monnr(j)));
    end
    curr_time = curr_datestr(end-7:end);
    
    nc = ones(20,1);
    
    for j = 1:length(imlink)
        fprintf(h,'%i\t%i\t%i\t%i\t%i\t%i\t%s\t%i\t%i\t%i\t%s\t%s\t%s\n',...
            imlink(j,1),imlink(j,2),imlink(j,3),str(i).choice(j),str(i).RT(j),nc(j),subject_id,ubercounter,j,NaN,str(i).Answer_gender{1},curr_date,curr_time);
%         a3(ct,:) = imlink(j,1:3);
%         ct = ct-1;
    end
    
end

fclose(h);

%% then run this on the small set of batches collected by Oli

save ubercounter.mat ubercounter

clear variables

load ubercounter.mat

load('tmp.mat')
rng(tmp)
key = randi(36,1,21);

base_dir = '/Users/hebart/Documents/projects/hebart/things1854_behav_large/new_raw_results5';
fnames_orig = dir(fullfile(base_dir,'*.csv'));
fnames_orig = {fnames_orig.name}';
fnames_orig = strcat(repmat({[base_dir '/']},length(fnames_orig),1),fnames_orig);

disp('Oli''s triplets:')

for i_name = 1:length(fnames_orig)
    
%     disp(i_name)
    
    filename = fnames_orig{i_name};

%% Initialize variables.
delimiter = ',';
startRow = 2;
endRow = inf;

%% Open the text file.
fileID = fopen(filename,'r');

%% figure out the number of columns
header = fgetl(fileID);
header = strrep(header,'.','_'); % replace . with _
header = strrep(header,'"',''); % remove ""
variable_names = strsplit(header,',');

frewind(fileID);

%% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = [repmat('%q',1,length(variable_names)) '%[^\n\r]'];

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

% batchstr = strsplit(dataArray{9}{1},';');
% batch_nr_orig(i,1) = str2double(batchstr{1}(9:end));

if length(dataArray)>length(variable_names)
    if ~all(cellfun(@isempty,dataArray{end})), error('mismatch between variable names and dataArray size'), end
    
    dataArray = dataArray(1:length(variable_names)); % remove empty column at back
    
end

% variables that we leave out
left_out = {'Approve','Reject','Answer_doNotRedirect'};

[~,rmind] = intersect(variable_names,left_out);

variable_names(rmind) = [];
dataArray(rmind) = [];

if length(dataArray) ~= length(variable_names)
    error('mismatch between variable names and dataArray size')
end

answer_field = find(~cellfun(@isempty,strfind(variable_names,'Answer')));

clear str
str(length(dataArray{1}),1).tmp = 0; % trick to initialize
for i = 1:length(variable_names)
    if any(answer_field==i), continue, end
    [str.(variable_names{i})] = deal(dataArray{i}{:});
end
str = rmfield(str,'tmp');

% gender, hispanic, race
ghr = {'gender','hispanic','race'};
for i = answer_field
    
    currind = ~cellfun(@isempty,regexp(variable_names{i},ghr));
    if any(currind)
        [str.(ghr{currind})] = deal(dataArray{i}{:});
    end
end

% THIS code is way faster ;)
tmp = num2cell(str2double({str.WorkTimeInSeconds}'),2);
[str.duration_in_seconds] = deal(tmp{:});


currind = strcmp(variable_names,'Answer_expDuration');
cf = cellfun(@str2double,dataArray{currind})/1000;
cf = num2cell(cf,2);
[str.exp_duration] = deal(cf{:});

in_names = {'Answer_RT','Answer_question','Answer_imLink1','Answer_imLink2','Answer_imLink3'};
out_names = {'RT','choice','imlink1','imlink2','imlink3'};

% format choices
currind = strcmp(variable_names,'Answer_question');
dataArray(currind) = cellfun(@(x) strrep(x,'link',''),dataArray(currind),'uniformoutput',0);

for i = 1:length(in_names)
currind = strcmp(variable_names,in_names{i});
cf = cellfun(@(x) sscanf(x,'%f')',dataArray{currind},'uniformoutput',0);
% cf = num2cell(cf,2);
[str.(out_names{i})] = deal(cf{:});
end

[str.batch_id] = deal(i_name);

STR{i_name,1} = str;

end

str = vertcat(STR{:});

n = length(STR);

clear STR

%% check for exact duplicates and remove


hit_id = {str.HITId}';
batch_id = {str.batch_id}';
WorkerId = {str.WorkerId}';

[~,ind] = unique(hit_id);

m = ones(length(hit_id),1);
m(ind) = 0;

m_ind = find(m);

% now let's see if they are just the same or exact duplicates using hit_id AND WorkerId:

exact_duplicate_ind = [];
for i = 1:length(m_ind)
    duplicate_ind = find(strcmp(hit_id,hit_id(m_ind(i))));
    id = WorkerId(duplicate_ind);
    if strcmp(id(1),id(2))
        exact_duplicate_ind(end+1) = duplicate_ind(2);
    end
end
    
str(exact_duplicate_ind) = [];
fprintf('Number of HITs/trials removed due to duplicates: %i / %i\n',length(exact_duplicate_ind),length(exact_duplicate_ind)*20)



%% check for cheaters

[rmind,n_rm,n_unique] = check_for_cheaters(str);

%% check performance on original model of removed triplets
% fprintf('Performance of removed triplets: %.2f\n',check_performance_orig(str(rmind)))


%% now remove those from str

str(rmind) = [];
% fprintf('Performance of remaining triplets: %.2f\n',check_performance_orig(str))
fprintf('Number of HITs/trials removed due to speed or repetitiveness: %i / %i\n',length(rmind),length(rmind)*20)
fprintf('Number of workers removed from this dataset: %i / %i\n',n_rm,n_unique)

%% 

% now check for HITs that we remove
rmindnan = false(length(str),1);
for i = 1:length(str)
    if any(isnan(str(i).RT)) || isempty(str(i).RT)
        rmindnan(i) = true;
    end
end
str(rmindnan) = [];
fprintf('Number of HITs/trials removed due to being empty: %i / %i\n',sum(rmindnan),sum(rmindnan)*20)
fprintf('Total number of HITs/trials included: %i / %i\n',length(str),length(str)*20)


%% Let's write main results out

% we want to keep triplet, choice, RT, noise ceiling, ID, HIT_nr, trial_nr, age, gender, date, time 

h = fopen('triplets_large_temp_part3.csv','w');
fprintf(h,'image1\timage2\timage3\tchoice\tRT\tnoise_ceiling\tsubject_id\tHIT_nr\ttrial_nr\tage\tgender\tdate\ttime\n');

% recode worker ids to subject ids
subject_ids = recode_workerids({str.WorkerId},key);

for i = 1:length(str)
    if any(isnan(str(i).choice))
        keyboard
    end
    
    subject_id = subject_ids{i};
    
    imlink = [str(i).imlink1' str(i).imlink2' str(i).imlink3'];
    
    ubercounter = ubercounter + 1;
    
    curr_datestr = str(i).SubmitTime;
    curr_datestr = strrep(curr_datestr,'PST ','');
    curr_datestr = strrep(curr_datestr,'PDT ','');
    curr_datestr = [curr_datestr(end-3:end) ' ' curr_datestr(5:end-5)];
    curr_date = strrep(curr_datestr(1:end-9),' ','-');
    % go through months
    mon = {'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'};
    monnr = 1:12;
    for j = 1:12
        curr_date = strrep(lower(curr_date),lower(mon{j}),sprintf('%02i',monnr(j)));
    end
    curr_time = curr_datestr(end-7:end);
    
    nc = zeros(20,1);
    
    for j = 1:length(imlink)
        fprintf(h,'%i\t%i\t%i\t%i\t%i\t%i\t%s\t%i\t%i\t%i\t%s\t%s\t%s\n',...
            imlink(j,1),imlink(j,2),imlink(j,3),str(i).choice(j),str(i).RT(j),nc(j),subject_id,ubercounter,j,NaN,str(i).gender,curr_date,curr_time);
    end
    
end

fclose(h);



%% Finally, run this on the new batches
save ubercounter.mat ubercounter

clear variables

load ubercounter.mat

load('tmp.mat')
rng(tmp)
key = randi(36,1,21);

base_dir = '/Users/hebart/Documents/projects/hebart/things1854_behav_large/new_raw_results2';
fnames_orig = dir(fullfile(base_dir,'*.csv'));
fnames_orig = {fnames_orig.name}';
fnames_orig = strcat(repmat({[base_dir '/']},length(fnames_orig),1),fnames_orig);

base_dir = '/Users/hebart/Documents/projects/hebart/things1854_behav_large/new_raw_results3';
fnames_orig2 = dir(fullfile(base_dir,'*.csv'));
fnames_orig2 = {fnames_orig2.name}';
fnames_orig2 = strcat(repmat({[base_dir '/']},length(fnames_orig2),1),fnames_orig2);

base_dir = '/Users/hebart/Documents/projects/hebart/things1854_behav_large/new_raw_results4';
fnames_orig3 = dir(fullfile(base_dir,'*.csv'));
fnames_orig3 = {fnames_orig3.name}';
fnames_orig3 = strcat(repmat({[base_dir '/']},length(fnames_orig3),1),fnames_orig3);

fnames_orig = [fnames_orig; fnames_orig2; fnames_orig3];

disp('New batches')

for i_name = 1:length(fnames_orig)
    
%     disp(i_name)
    
    filename = fnames_orig{i_name};

%% Initialize variables.
delimiter = ',';
startRow = 2;
endRow = inf;

%% Open the text file.
fileID = fopen(filename,'r');

%% figure out the number of columns
header = fgetl(fileID);
header = strrep(header,'.','_'); % replace . with _
header = strrep(header,'"',''); % remove ""
variable_names = strsplit(header,',');

frewind(fileID);

%% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = [repmat('%q',1,length(variable_names)) '%[^\n\r]'];

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

% batchstr = strsplit(dataArray{9}{1},';');
% batch_nr_orig(i,1) = str2double(batchstr{1}(9:end));

if length(dataArray)>length(variable_names)
    if ~all(cellfun(@isempty,dataArray{end})), error('mismatch between variable names and dataArray size'), end
    
    dataArray = dataArray(1:length(variable_names)); % remove empty column at back
    
end

% variables that we leave out
left_out = {'Approve','Reject','Answer_doNotRedirect'};

[~,rmind] = intersect(variable_names,left_out);

variable_names(rmind) = [];
dataArray(rmind) = [];

if length(dataArray) ~= length(variable_names)
    error('mismatch between variable names and dataArray size')
end

answer_field = find(~cellfun(@isempty,strfind(variable_names,'Answer')));

clear str
str(length(dataArray{1}),1).tmp = 0; % trick to initialize
for i = 1:length(variable_names)
    if any(answer_field==i), continue, end
    [str.(variable_names{i})] = deal(dataArray{i}{:});
end
str = rmfield(str,'tmp');

% gender, hispanic, race, age
ghr = {'gender','hispanic','race','age'};
for i = answer_field
    
    currind = ~cellfun(@isempty,regexp(variable_names{i},ghr));
    if any(currind)
        [str.(ghr{currind})] = deal(dataArray{i}{:});
    end
end

tmp = num2cell(str2double({str.WorkTimeInSeconds}'),2);
[str.duration_in_seconds] = deal(tmp{:});


currind = strcmp(variable_names,'Answer_expDuration');
cf = cellfun(@str2double,dataArray{currind})/1000;
cf = num2cell(cf,2);
[str.exp_duration] = deal(cf{:});

in_names = {'Answer_RT','Answer_question','Answer_imLink1','Answer_imLink2','Answer_imLink3'};
out_names = {'RT','choice','imlink1','imlink2','imlink3'};

% format choices
currind = strcmp(variable_names,'Answer_question');
dataArray(currind) = cellfun(@(x) strrep(x,'link',''),dataArray(currind),'uniformoutput',0);

for i = 1:length(in_names)
currind = strcmp(variable_names,in_names{i});
cf = cellfun(@(x) sscanf(x,'%f')',dataArray{currind},'uniformoutput',0);
% cf = num2cell(cf,2);
[str.(out_names{i})] = deal(cf{:});
end

[str.batch_id] = deal(i_name);

STR{i_name,1} = str;

end

str = vertcat(STR{:});

n = length(STR);

clear STR

%% check for exact duplicates and remove


hit_id = {str.HITId}';
batch_id = {str.batch_id}';
WorkerId = {str.WorkerId}';

[~,ind] = unique(hit_id);

m = ones(length(hit_id),1);
m(ind) = 0;

m_ind = find(m);

% now let's see if they are just the same or exact duplicates using hit_id AND WorkerId:

exact_duplicate_ind = zeros(size(m_ind));
cnt = 0;
for i = 1:length(m_ind)
    if ~mod(i,1000), disp(m_ind), end
    duplicate_ind = find(strcmp(hit_id,hit_id(m_ind(i))));
    id = WorkerId(duplicate_ind);
    if strcmp(id(1),id(2))
        cnt = cnt+1;
        exact_duplicate_ind(cnt) = duplicate_ind(2);
    end
end
    
str(exact_duplicate_ind) = [];
fprintf('Number of HITs/trials removed due to duplicates: %i / %i\n',length(exact_duplicate_ind),length(exact_duplicate_ind)*20)



%% check for cheaters
n_rm = 0; n_unique = 0; rmind = [];

[rmind,n_rm,n_unique] = check_for_cheaters(str);

%% check performance on original model of removed triplets
% fprintf('Performance of removed triplets: %.2f\n',check_performance_orig(str(rmind)))


%% now remove those from str

str(rmind) = [];
% fprintf('Performance of remaining triplets: %.2f\n',check_performance_orig(str))
fprintf('Number of HITs/trials removed due to speed or repetitiveness: %i / %i\n',length(rmind),length(rmind)*20)
fprintf('Number of workers removed from this dataset: %i / %i\n',n_rm,n_unique)

%% 

% now check for HITs that we remove
rmindnan = false(length(str),1);
for i = 1:length(str)
    if any(isnan(str(i).RT)) || isempty(str(i).RT)
        rmindnan(i) = true;
    end
end
str(rmindnan) = [];
fprintf('Number of HITs/trials removed due to being empty: %i / %i\n',sum(rmindnan),sum(rmindnan)*20)
fprintf('Total number of HITs/trials included: %i / %i\n',length(str),length(str)*20)


% count how often workers participated
allWorkerIds = {str.WorkerId}';
uniqueIds = unique(allWorkerIds);
clear nn
for i = 1:length(uniqueIds)
    nn(i) = sum(strcmpi(allWorkerIds,uniqueIds{i}));
end
% return all workers who did more than 350 HITs and mark them separately
toomanyHITsIds = uniqueIds(nn>=350);

%% Let's write main results out

% we want to keep triplet, choice, RT, noise ceiling, ID, HIT_nr, trial_nr, age, gender, date, time 

h = fopen('triplets_large_temp_part4.csv','w');
fprintf(h,'image1\timage2\timage3\tchoice\tRT\tnoise_ceiling\tsubject_id\tHIT_nr\ttrial_nr\tage\tgender\tdate\ttime\n');

% recode worker ids to subject ids
subject_ids = recode_workerids({str.WorkerId},key);

for i = 1:length(str)
    if any(isnan(str(i).choice))
        keyboard
    end
    
    subject_id = subject_ids{i};
    
    imlink = [str(i).imlink1' str(i).imlink2' str(i).imlink3'];
    
    ubercounter = ubercounter + 1;
    
    curr_datestr = str(i).SubmitTime;
    curr_datestr = strrep(curr_datestr,'PST ','');
    curr_datestr = strrep(curr_datestr,'PDT ','');
    curr_datestr = [curr_datestr(end-3:end) ' ' curr_datestr(5:end-5)];
    curr_date = strrep(curr_datestr(1:end-9),' ','-');
    % go through months
    mon = {'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'};
    monnr = 1:12;
    for j = 1:12
        curr_date = strrep(lower(curr_date),lower(mon{j}),sprintf('%02i',monnr(j)));
    end
    curr_time = curr_datestr(end-7:end);
    
    nc = str2double(strsplit(str(i).Input_noise_ceiling));
    nc(nc>0) = nc(nc>0)+1;
    
    for j = 1:length(imlink)
        fprintf(h,'%i\t%i\t%i\t%i\t%i\t%i\t%s\t%i\t%i\t%i\t%s\t%s\t%s\n',...
            imlink(j,1),imlink(j,2),imlink(j,3),str(i).choice(j),str(i).RT(j),nc(j),subject_id,ubercounter,j,str2double(str(i).age),str(i).gender,curr_date,curr_time);
%         a3(ct,:) = imlink(j,1:3);
%         ct = ct-1;
    end
    
end

fclose(h);

delete tmp.mat
delete ubercounter.mat

%% combine datasets

clear variables

tab1 = readtable('triplets_large_temp_part1.csv');
tab1.dataset = 1*ones(height(tab1),1);
tab2 = readtable('triplets_large_temp_part2.csv');
tab2.dataset = 2*ones(height(tab2),1);
tab3 = readtable('triplets_large_temp_part3.csv');
tab3.dataset = 3*ones(height(tab3),1);
tab4 = readtable('triplets_large_temp_part4.csv');
tab4.dataset = 4*ones(height(tab4),1);
tab = [tab1;tab2;tab3;tab4];
clear tab1 tab2 tab3 tab4

% before writing the final table, let's remove workers with very
% inconsistent responses in age or gender (because this turns out to be
% predictive of their performance)

%% run basic stats on combined dataset

% tab = readtable('triplets_large_final_variant2_corrected.csv');

% now go through all workers and find their most commonly reported demographics
% luckily, we only need to check every 20 rows

[u_id,~,uia] = unique(tab{:,7});
n_workers_total = length(u_id);

age_all = tab{:,10};

ages = cell(n_workers_total,1);
genders = cell(n_workers_total,1);
for i = 1:20:height(tab)
%     ind = strcmp(u_id,tab{i,7});
    ind = uia(i);
    ages{ind} = [ages{ind} age_all(i)];
    genders{ind}(end+1) = tab{i,11};
end

% now pick most common answer
for i = 1:length(ages)
    tmp = ages{i};
    % convert ages with birtdate to assumed collection in 2020
    tmp(tmp>1900) = 2020-tmp(tmp>1900);
    tmp(tmp<18) = 18;
    utmp = unique(tmp);
    if any(isnan(utmp)), utmp(isnan(utmp)) = []; utmp(end+1) = NaN; end
    clear acnt
    for j = length(utmp):-1:1
        acnt(j) = sum(tmp==utmp(j));
    end
    [~,ii] = max(acnt);
    final_ages(i,1) = utmp(ii);
    n_ages(i,1) = length(utmp);
    sd_ages(i,1) = nanstd(tmp);
    
end

clear final_gender
for i = 1:length(genders)
    tmp = genders{i};
    % convert ages with birtdate to assumed collection in 2020
    
    utmp = unique(tmp);
    clear acnt
    for j = length(utmp):-1:1
        acnt(j) = sum(strcmp(tmp,utmp(j)));
    end
    [~,ii] = max(acnt);
    final_gender(i,1) = utmp(ii);
    n_genders(i,1) = length(utmp);
    
end

clear age_all ages genders

% let's remove all workers with more than 1 age and more than 1 gender, or with more than 3 ages
rmind = find(n_genders==2 & n_ages>1);
rmind2 = find(n_ages>3);

% we only write the latter, removing the former would mean 500k entries,
% which is excessive
tab(ismember(uia,rmind2),:) = [];


writetable(tab,'triplets_large_final.csv');

%% correct noiseceiling
clear tab
disp('Correcting noise ceilings...')
tab = correct_noiseceiling('triplets_large_final.csv','triplets_large_final_corrected.csv');




%% Let's check noise ceiling

% tab = readtable('triplets_large_final_correctednc_correctedorder.csv');

nc1 = tab.noise_ceiling==1; % original noise ceiling (1000 random triplets)
nc2 = tab.noise_ceiling==2; % new within subject noise ceiling (another 1000)
nc3 = tab.noise_ceiling==3; % new between subject noise ceiling (same 1000 as nc2)

%

% compute noise ceiling
nc1_triplets = table2array(tab(nc1,1:4));
[nc1m,nc1_95CI] = get_noiseceiling(nc1_triplets);

nc2_triplets = table2array(tab(nc2,1:4));
nc2_hitid = table2array(tab(nc2,8));
nc2_trialid = table2array(tab(nc2,9));
% remove triple entry
% nc2_triplets(5089,:) = [];

% todo: sort triplets correctly
[nc2m_within,nc2_within_95CI,nc2m_between,nc2_between_95CI] = get_noiseceiling_within(nc2_triplets,nc2_hitid,nc2_trialid);

nc3_triplets = table2array(tab(nc3,1:4));
[nc3m,nc3_95CI] = get_noiseceiling(nc3_triplets);

%% now sort to correct order

if ~exist('tab','var')
    tab = readtable('triplets_large_final_corrected.csv');
end
triplets = tab{:,1:3};
% sort to correct order
load(fullfile('/Users/hebart/Documents/projects/hebart/nathumbehav/variables','sortind.mat'));
for i_obj = 1:1854
    triplets(triplets==sortind(i_obj)) = 10000+i_obj;
end
triplets = triplets-10000;
tab{:,1:3} = triplets;
writetable(tab,'triplets_large_final_correctednc_correctedorder.csv','delimiter','\t')

%% now extract all test sets and write them out, plus remove

x = 1;
if x

clear D
[D(:,1),D(:,2),D(:,3),D(:,4),D(:,5),D(:,6)] = textread('triplets_large_final_correctednc_correctedorder.csv','%n%n%n%n%n%n%*[^\n]','delimiter','\t','emptyvalue',NaN,'headerlines',1,'bufsize',50000);

% make everything 0 base
D(:,1:3) = D(:,1:3)-1;


D(D(:,4)==1,1:3) = D(D(:,4)==1,[2 3 1]);
D(D(:,4)==2,1:3) = D(D(:,4)==2,[1 3 2]);
% third is just stay as it is

testset1 = D(D(:,6)==1,:);
testset2 = D(D(:,6)==2,:);
testset3 = D(D(:,6)==3,:);


h = fopen('testset1.txt','w');
for i = 1:length(testset1)
    currtrial = testset1(i,:);
    fprintf(h,'%i %i %i\n',[currtrial(1) currtrial(2) currtrial(3)]);
end
fclose(h);

h = fopen('testset2.txt','w');
h2 = fopen('testset2_repeat.txt','w');
cnt = 0;
for i = 1:2:length(testset2)
    cnt = cnt+1;
    currtrial = testset2(i,:);
    fprintf(h,'%i %i %i\n',[currtrial(1) currtrial(2) currtrial(3)]);
    currtrial = testset2(i+1,:);
    fprintf(h2,'%i %i %i\n',[currtrial(1) currtrial(2) currtrial(3)]);
end
fclose(h);
fclose(h2);

h = fopen('testset3.txt','w');
for i = 1:length(testset3)
    currtrial = testset3(i,:);
    fprintf(h,'%i %i %i\n',[currtrial(1) currtrial(2) currtrial(3)]);
end
fclose(h);

%% then write out every 10th trial and make that the validation set, plus remove
D2 = D;
D2(D2(:,6)>0,:) = [];

validation_ind = (10:10:length(D2))';
rng(1)
validation_ind = validation_ind + randi(11,length(validation_ind),1)-6;
validation_ind = unique(validation_ind);
validationset = D2(validation_ind,:);

h = fopen('validationset.txt','w');
for i = 1:length(validationset)
   currtrial = validationset(i,:);
   fprintf(h,'%i %i %i\n',[currtrial(1) currtrial(2) currtrial(3)]);
end
fclose(h);

%% then write out all other data
D2(validation_ind,:) = [];
trainset = D2;

h = fopen('trainset.txt','w');
for i = 1:length(trainset)
   currtrial = trainset(i,:);
   fprintf(h,'%i %i %i\n',[currtrial(1) currtrial(2) currtrial(3)]);
end
fclose(h);

end