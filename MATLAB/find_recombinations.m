function A = find_recombinations(A)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

n = length(A);

% Rows of triolist correspond to parent-child trios s.t. the child has no
% intervening parent
trioList = zeros(nnz(A),7);

% Output edge list
edgeList = zeros(nnz(A),3);

% Build the triolist
nTrio = 1;
nEdge = 0;
B = A';
for child = 1:n
    [rowParents, ~, rowWeights] = find(B(:,child));
    if ~isempty(rowWeights)
        [~, ~, weightInd] = unique(rowWeights);
        weight_sets = index_array(weightInd);
        % weight_sets = {find(rowWeights < 0) find(rowWeights > 0)}
        for kk = 1:length(weight_sets)
            if ~isempty(weight_sets{kk})
                rows = cell2mat(weight_sets{kk}.keys);
                these_parents = rowParents(rows);
                if length(these_parents) > 1
                    for pp = 2:length(these_parents)
                        nTrio = nTrio+1;
                        trioList(nTrio,1:4) = [these_parents(pp-1) these_parents(pp) ...
                            child rowWeights(rows(pp)) ];
                        
                        % Pointers to rows of trioList that share an edge
                        if pp > 2
                            trioList(nTrio, 6) = nTrio-1;
                            trioList(nTrio - 1, 7) = nTrio;
                        end
                    end
                else
                    nEdge = nEdge+1;
                    edgeList(nEdge,:) = [these_parents child rowWeights(rows)];
                end
            end
        end
    end
end
trioList = trioList(1:nTrio+1,:);

[cliques, ~, which_clique] = unique(trioList(1:end-1,1:2),'rows');
nClique = size(cliques,1);
cliques(end+1:nTrio,:) = 0;
cliqueRows = index_array(which_clique, nTrio);
cliqueSize = cellfun(@length,cliqueRows);
trioList(1:end-1,5) = which_clique;

newNode = n;
loopcounter = 0;
while true
    loopcounter = loopcounter + 1;
    [nr, c] = max(cliqueSize);
    
    rows = cell2mat(cliqueRows{c}.keys);
    rowParents = cliques(c,:);
    rowChildren = trioList(rows, 3);
    rowWeights = trioList(rows, 4);
    
    % No nontrivial cliques to be collapsed
    if nr <= 1
        break
    end    
    
    % Introduce a new node
    newNode = newNode + 1;

    % Update trio list for the rows such that parents(p) is
    % parent(3-p) of that row
    has_shared_duo = cell(2,1);
    for p = 1:2
        [nClique, has_shared_duo{p}] = update_trioList(p, rows, newNode, nClique);
    end
    
    % Children with no shared duos can be added to the edge list
    noSharedDuos = ~ (has_shared_duo{1} | has_shared_duo{2});
    edgeList(nEdge+1:nEdge+sum(noSharedDuos),1) = newNode;
    edgeList(nEdge+1:nEdge+sum(noSharedDuos),2) = rowChildren(noSharedDuos);
    edgeList(nEdge+1:nEdge+sum(noSharedDuos),3) = rowWeights(noSharedDuos);
    nEdge = nEdge+sum(noSharedDuos);

    % For children that have two shared duos, they need to be "patched
    % together": their left-pointers and right-pointers, which currently
    % point to a row in the current clique, point to each other instead
    twoSharedDuos = has_shared_duo{1} & has_shared_duo{2};
    adjacentRows = cell(2,1);
    for p = 1:2
        adjacentRows{p} = trioList(rows(twoSharedDuos), 5+p);
    end
    for p = 1:2
        trioList(adjacentRows{3-p}, 5+p) = adjacentRows{p};
    end

    % For children that have exactly one shared duo, the corresponding
    % left- or right-pointers need to be zeroed out
    for p = 1:2
        oneSharedDuo = has_shared_duo{p} &~ has_shared_duo{3-p};
        adjacentRows = trioList(rows(oneSharedDuo), 5+p);
        trioList(adjacentRows,5+(3-p)) = 0;
    end
    
    % Replace current clique with a singleton clique
    trioList(rows(1), :) = [rowParents newNode 1 c 0 0];
    trioList(rows(2:end), :) = 0;
    cliqueRows{c} = containers.Map(rows(1), true);
    cliqueSize(c) = 1;
end

% Remaining trios
trioList = trioList(trioList(:,3) ~=0, 1:4);
nTrio =  size(trioList,1);
for p = 1:2
    edgeList(nEdge+1:nEdge+nTrio,:) = trioList(:, [p 3 4]);
    nEdge = nEdge + nTrio;
end
edgeList = edgeList(1:nEdge,:);

% Remove redundant edges
edgeList = unique(edgeList, 'rows');

% Form sparse matrix
A = sparse(edgeList(:,2),edgeList(:,1),edgeList(:,3),newNode,newNode);

% Returns a cell array of length max(idx) [or n] such that all(idx(cells{j}) == j)
    function cells = index_array(idx,n)
        if isempty(idx)
            cells = containers.Map;
            return;
        end
        if ~exist('n')
            n = max(idx);
        end
        cells = accumarray(idx,(1:length(idx))',[n 1],@(x){containers.Map(x,true(size(x)))});
        cells(cellfun(@isempty,cells)) = repmat({containers.Map},sum(cellfun(@isempty,cells)),1);
    end

% Updates rows of the trio list sharing a duo with the rows of the current
% clique. For each such row, its parent is updated to newNode; for each
% affected clique, a new clique is created; the number of cliques is also 
% updated. Returns the new number of cliques.
    function [nClique, has_shared_duo] = update_trioList(p, rows, newNode, nClique)
        nn = length(rows);
        if nn == 0
            return
        end

        % Trios that need to be updated
        has_shared_duo = trioList(rows, 5+p) ~= 0;
        whichRows = trioList(rows(has_shared_duo), 5+p) ;
        
        % Update parent 3-p of each trio that needs to be updated
        trioList(whichRows, 3 - p) = newNode;
        
        % Cliques that need to be updated
        [affected_cliques, ~, which_affected_cliques] = unique(trioList(whichRows,5));
        
        % Create a new clique for each such clique + assign affected rows
        % to the new clique
        for wc = 1:length(affected_cliques)
            clique_rows = whichRows(which_affected_cliques == wc);
            cliqueRows{affected_cliques(wc)} = remove(cliqueRows{affected_cliques(wc)}, ...
                num2cell(clique_rows));
            cliqueSize(affected_cliques(wc)) = length(cliqueRows{affected_cliques(wc)});

            cliqueRows{nClique + 1} = containers.Map(clique_rows,...
                true(size(clique_rows)));
            cliqueSize(nClique + 1) = length(cliqueRows{nClique + 1});
            cliques(nClique + 1, :) = trioList(clique_rows(1), 1:2);
            trioList(clique_rows,5) = nClique + 1;
            nClique = nClique + 1;
        end
    end

end