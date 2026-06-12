function out = compare_model_rankings(id_SFA, id_classic, prob_SFA, BIC_SFA, BIC_classic, topN)

%COMPARE_MODEL_RANKINGS
% Compares top model rankings using:
%   1. SFA posterior model probability
%   2. SFA BIC
%   3. classic BMA BIC
%
% Inputs:
%   id_SFA       - model IDs from SFA-BMA
%   id_classic   - model IDs from classic BMA
%   prob_SFA     - posterior model probabilities from SFA-BMA
%   BIC_SFA      - BIC values from SFA-BMA
%   BIC_classic  - BIC values from classic BMA
%   topN         - number of top models to compare

if nargin < 6
    topN = 100;
end

% Force column vectors
id_SFA      = id_SFA(:);
id_classic  = id_classic(:);
prob_SFA    = prob_SFA(:);
BIC_SFA     = BIC_SFA(:);
BIC_classic = BIC_classic(:);

% Build tables
SFA = table(id_SFA, prob_SFA, BIC_SFA, 'VariableNames', {'model_id','prob_SFA','BIC_SFA'});

CLASSIC = table(id_classic, BIC_classic, 'VariableNames', {'model_id','BIC_classic'});

% Rankings
% SFA probability: higher better
[~, ord_prob] = sort(SFA.prob_SFA, 'descend');
SFA.rank_prob = zeros(height(SFA),1);
SFA.rank_prob(ord_prob) = (1:height(SFA))';

% SFA BIC: lower better
[~, ord_bic_sfa] = sort(SFA.BIC_SFA, 'ascend');
SFA.rank_BIC_SFA = zeros(height(SFA),1);
SFA.rank_BIC_SFA(ord_bic_sfa) = (1:height(SFA))';

% classic BIC: lower better
[~, ord_bic_classic] = sort(CLASSIC.BIC_classic, 'ascend');
CLASSIC.rank_BIC_classic = zeros(height(CLASSIC),1);
CLASSIC.rank_BIC_classic(ord_bic_classic) = (1:height(CLASSIC))';

% Join by model ID
C = innerjoin(SFA, CLASSIC, 'Keys', 'model_id');

% Adjust topN
topN_sfa     = min(topN, height(SFA));
topN_classic = min(topN, height(CLASSIC));

% TopN sets

top_prob_ids = SFA.model_id(SFA.rank_prob <= topN_sfa);
top_bic_sfa_ids = SFA.model_id(SFA.rank_BIC_SFA <= topN_sfa);
top_bic_classic_ids = CLASSIC.model_id(CLASSIC.rank_BIC_classic <= topN_classic);

% Overlap sets

overlap_prob_BIC_SFA_ids = intersect(top_prob_ids, top_bic_sfa_ids);
overlap_prob_BIC_classic_ids = intersect(top_prob_ids, top_bic_classic_ids);
overlap_BIC_SFA_BIC_classic_ids = intersect(top_bic_sfa_ids, top_bic_classic_ids);

% Output
out = struct();
out.topN = topN;

% Overlap rates
out.overlap_prob_vs_BIC_SFA = numel(overlap_prob_BIC_SFA_ids) / topN_sfa;
out.overlap_prob_vs_BIC_classic = numel(overlap_prob_BIC_classic_ids) / topN_sfa;
out.overlap_BIC_SFA_vs_BIC_classic = numel(overlap_BIC_SFA_BIC_classic_ids) / topN_sfa;

% Same topN sets (not really that interesting)
out.same_topN_prob_vs_BIC_SFA = isequal(sort(top_prob_ids), sort(top_bic_sfa_ids));
out.same_topN_prob_vs_BIC_classic = isequal(sort(top_prob_ids), sort(top_bic_classic_ids));
out.same_topN_BIC_SFA_vs_BIC_classic = isequal(sort(top_bic_sfa_ids), sort(top_bic_classic_ids));

% Probability mass in overlaps
top_prob_mass = sum(SFA.prob_SFA(ismember(SFA.model_id, top_prob_ids)), 'omitnan');

out.sum_prob_overlap_prob_BIC_SFA = sum(SFA.prob_SFA(ismember(SFA.model_id, overlap_prob_BIC_SFA_ids)), 'omitnan');
out.sum_prob_overlap_prob_BIC_classic = sum(SFA.prob_SFA(ismember(SFA.model_id, overlap_prob_BIC_classic_ids)), 'omitnan');
out.sum_prob_overlap_BIC_SFA_BIC_classic = sum(SFA.prob_SFA(ismember(SFA.model_id, overlap_BIC_SFA_BIC_classic_ids)), 'omitnan');

if top_prob_mass > 0
    out.share_prob_overlap_prob_BIC_SFA = out.sum_prob_overlap_prob_BIC_SFA / top_prob_mass;

    out.share_prob_overlap_prob_BIC_classic = out.sum_prob_overlap_prob_BIC_classic / top_prob_mass;
else
    out.share_prob_overlap_prob_BIC_SFA = NaN;
    out.share_prob_overlap_prob_BIC_classic = NaN;
end

% Rank correlations

if height(C) > 1
    out.spearman_prob_vs_BIC_SFA = corr(C.rank_prob, C.rank_BIC_SFA, 'Type','Spearman','Rows','complete');
    out.spearman_prob_vs_BIC_classic = corr(C.rank_prob, C.rank_BIC_classic, 'Type','Spearman','Rows','complete');
    out.spearman_BIC_SFA_vs_BIC_classic = corr(C.rank_BIC_SFA, C.rank_BIC_classic, 'Type','Spearman','Rows','complete');
else
    out.spearman_prob_vs_BIC_SFA = NaN;
    out.spearman_prob_vs_BIC_classic = NaN;
    out.spearman_BIC_SFA_vs_BIC_classic = NaN;
end

% Diagnostics table

D = C(ismember(C.model_id, top_prob_ids), :);
D = sortrows(D, 'rank_prob');

out.top_prob_table = D(:, { ...
    'model_id', ...
    'prob_SFA', ...
    'BIC_SFA', ...
    'BIC_classic', ...
    'rank_prob', ...
    'rank_BIC_SFA', ...
    'rank_BIC_classic'});

out.joined_table = C;

% Summary table

out.summary = table( ...
    out.overlap_prob_vs_BIC_SFA, ...
    out.overlap_prob_vs_BIC_classic, ...
    out.overlap_BIC_SFA_vs_BIC_classic, ...
    out.sum_prob_overlap_prob_BIC_SFA, ...
    out.sum_prob_overlap_prob_BIC_classic, ...
    out.sum_prob_overlap_BIC_SFA_BIC_classic, ...
    out.share_prob_overlap_prob_BIC_SFA, ...
    out.share_prob_overlap_prob_BIC_classic, ...
    out.spearman_prob_vs_BIC_SFA, ...
    out.spearman_prob_vs_BIC_classic, ...
    out.spearman_BIC_SFA_vs_BIC_classic, ...
    'VariableNames', { ...
    'overlap_prob_BIC_SFA', ...
    'overlap_prob_BIC_classic', ...
    'overlap_BIC_SFA_BIC_classic', ...
    'sum_prob_overlap_prob_BIC_SFA', ...
    'sum_prob_overlap_prob_BIC_classic', ...
    'sum_prob_overlap_BIC_SFA_BIC_classic', ...
    'share_prob_overlap_prob_BIC_SFA', ...
    'share_prob_overlap_prob_BIC_classic', ...
    'rho_prob_BIC_SFA', ...
    'rho_prob_BIC_classic', ...
    'rho_BIC_SFA_BIC_classic'});
end