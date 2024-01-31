function A = find_recombinations(A)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

n = length(A);

% Rows of triolist correspond to parent-child trios s.t. the child has no
% intervening parent
trioList = zeros(nnz(A),4);

% Output edge list
edgeList = zeros(nnz(A),3);

% Build the triolist
nTrio = 0;
nEdge = 0;
for child = 1:n
    [~, rowParents, rowWeights] = find(A(child,:));
    if ~isempty(rowWeights)
        [~, ~, weightInd] = unique(rowWeights);
        weight_sets = index_array(weightInd);
        % weight_sets = {find(rowWeights < 0) find(rowWeights > 0)}
        for kk = 1:length(weight_sets)
            if ~isempty(weight_sets{kk})
                these_parents = rowParents(weight_sets{kk});
                if length(these_parents) > 1
                    for pp = 2:length(these_parents)
                        nTrio = nTrio+1;

                        % Need to modify triolist so that it keeps track of both weights
                        trioList(nTrio,:) = [these_parents(pp-1) these_parents(pp) child rowWeights(weight_sets{kk}(pp)) ];
                    end
                else
                    nEdge = nEdge+1;
                    edgeList(nEdge,:) = [these_parents child rowWeights(weight_sets{kk})];
                end
            end
        end
    end
end
trioList = trioList(1:nTrio,:);

trioList = sortrows(trioList);
[cliques, ~, which_clique] = unique(trioList(:,1:2),'rows');
trioList(:,5) = which_clique;
nClique = size(cliques,1);
cliques = [cliques; zeros(length(trioList), 2)];

% Lookup tables for the rows of trio list
parentRows{1} = index_array(trioList(:,1), n);
parentRows{2} = index_array(trioList(:,2), n);
childRows = index_array(trioList(:,3), n);
cliqueRows = index_array(trioList(:,5), nClique);
cliqueRows(end+1:size(cliques,1)) = {[]};
cliqueSize = cellfun(@length,cliqueRows);

newNode = n;
loopcounter = 0;
while true
    loopcounter = loopcounter + 1;
    [nr, c] = max(cliqueSize);

    rows = cliqueRows{c};
    rowParents = cliques(c,:);
    rowChildren = trioList(rows, 3);
    rowWeights = trioList(rows, 4);
    
    % No nontrivial cliques to be collapsed
    if nr <= 1
        break
    end

    % Remove these rows from lookup tables
    parentRows{1}{rowParents(1)} = setdiff(parentRows{1}{rowParents(1)}, rows);
    parentRows{2}{rowParents(2)} = setdiff(parentRows{2}{rowParents(2)}, rows);
    for ch = 1:nr
        childRows{rowChildren(ch)} = setdiff(childRows{rowChildren(ch)}, rows);
    end
    cliqueRows{c} = [];
    cliqueSize(c) = 0;
    
    % Introduce a new node
    newNode = newNode + 1;

    % Connect parents with newNode
    edgeList(nEdge+1:nEdge+2,1) = rowParents;
    edgeList(nEdge+1:nEdge+2,2) = newNode;
    edgeList(nEdge+1:nEdge+2,3) = 1;
    nEdge = nEdge+2;

    % Connect newNode with any child that now has only one parent
    nParentPairs = cellfun(@(rows,weight)sum(trioList(rows,4)==weight),...
        childRows(rowChildren),num2cell(rowWeights));
    ch = find(nParentPairs == 0);
    edgeList(nEdge+1:nEdge+length(ch),1) = newNode;
    edgeList(nEdge+1:nEdge+length(ch),2) = rowChildren(ch);
    edgeList(nEdge+1:nEdge+length(ch),3) = rowWeights(ch);
    nEdge = nEdge+length(ch);

    % Update trio list for the rows such that parents(1) is
    % parent2 of that row
    nClique = update_trioList(2, rowParents(1), rowChildren, newNode, nClique);

    % Update trio list for the rows such that parents(2) is
    % parent1 of that row
    nClique = update_trioList(1, rowParents(2), rowChildren, newNode, nClique);

end

% Put trivial cliques into the edgelist
for child = 1:n
    rows = childRows{child};
    if isempty(rows)
        continue;
    end
    rowParents = trioList(rows, 1:2);
    rowWeights = repmat(trioList(rows, 4),1,2); % Same for both parents

    % Avoid double-counting edges
    [~, representatives] = unique(rowParents);
    edgeList(nEdge+1:nEdge+length(representatives),1) = rowParents(representatives);
    edgeList(nEdge+1:nEdge+length(representatives),2) = child;
    edgeList(nEdge+1:nEdge+length(representatives),3) = rowWeights(representatives);
    nEdge = nEdge+length(representatives);
end

edgeList = edgeList(1:nEdge,:);
A = sparse(edgeList(:,2),edgeList(:,1),edgeList(:,3),newNode,newNode);

% Returns a cell array of length max(idx) [or n] such that all(idx(cells{j}) == j)
    function cells = index_array(idx,n)
        if isempty(idx)
            cells = {};
            return;
        end
        if ~exist('n')
            n = max(idx);
        end
        cells = accumarray(idx,(1:length(idx))',[n 1],@(x){x});
    end

% Updates rows of the trio list of which one of its parents (depending
% on whichParent) equals parent, and of which its child is among
% children. Its parent is updated to newNode, its clique assignment is
% updated accordingly, the number of cliques is also updated. Returns
% the new number of cliques.
    function nClique = update_trioList(whichParent, parent, children, newNode, nClique)
        whichRows = intersect(vertcat(childRows{children}), ...
            parentRows{whichParent}{parent});
        parentRows{whichParent}{parent} = setdiff(parentRows{whichParent}{parent}, whichRows);
        trioList(whichRows, whichParent) = newNode;
        parentRows{whichParent}{newNode} = whichRows;

        [affected_cliques, ~, which_affected_cliques] = unique(trioList(whichRows,5));
        for wc = 1:length(affected_cliques)
            cliqueRows{affected_cliques(wc)} = setdiff(cliqueRows{affected_cliques(wc)}, ...
                whichRows);
            cliqueSize(affected_cliques(wc)) = length(cliqueRows{affected_cliques(wc)});
            cliqueRows{nClique + 1} = whichRows(which_affected_cliques == wc);
            cliqueSize(nClique + 1) = length(cliqueRows{nClique + 1});
            cliques(nClique + 1, :) = trioList(cliqueRows{nClique + 1}(1), 1:2);
            trioList(cliqueRows{nClique + 1},5) = nClique + 1;
            nClique = nClique + 1;
        end
    end

end