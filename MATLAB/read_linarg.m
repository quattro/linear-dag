function [A, samples, mutations] = read_linarg(filename, which_files)
%Read a linear ARG from a trio of files or a directory of them

if isfolder(filename)
    files = dir([filename,'/*.mtx']);
    counter = 0;
    if ~exist('which_files')
        which_files = 1:length(files);
    end
    nf = length(which_files);
    A = cell(nf,1); samples = A; mutations = A;

    for f = which_files
        [A{f}, samples{f}, mutations{f}] = read_linarg([filename, '/', files(f).name(1:end-4)]);
        samples{f}.index = samples{f}.index + counter;
        mutations{f}.index = mutations{f}.index + counter;
        counter = counter + length(A{f});
    end
    nn = cellfun(@length,A);
    nm = cellfun(@height,mutations);
    ns = cellfun(@height,samples);
    assert(all(ns==ns(1)))

    A = blkdiag(A{:});
    mutations = vertcat(mutations{:});
    samples = vertcat(samples{:});
    sample_indices = samples.index;

    % Aggregation by sample
    new_sample_indices = repelem((1:ns)', nf);
    non_sample_indices = setdiff((1:sum(nn))',sample_indices);
    new_non_sample_indices = non_sample_indices - repelem(ns(1)*(0:nf-1)',nn-ns);
    S = sparse([new_sample_indices; new_non_sample_indices], [sample_indices; non_sample_indices], 1);
    A = S * A * S';
    samples = samples(1:ns,:);
    samples.index = (1:ns)';
    [new_mutation_indices, columns] = find(S(:,mutations.index));
    assert(all(columns==(1:sum(nm))'))
    mutations.index = new_mutation_indices;

elseif isfile([filename, '.mtx'])
    A = readMatrixMarket([filename, '.mtx']);
    samples = readtable([filename, '.psam'], 'filetype', 'text');
    mutations = readtable([filename, '.pvar'], 'filetype', 'text');
    mutations.IDX = cellfun(@(x)sscanf(x, 'IDX=%d'), mutations.INFO);
else
    error('File not found')
end