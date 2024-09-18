% Visualize the dataset.
data = readtable('winequality-red.csv', 'VariableNamingRule', 'preserve'); % readtable reads data from csv file into a table. column represents variable(feature), row represents observation.
figure; % opens new window to display plots
for i = 1:width(data)-1 % loops through each column and plots a histogram to show the distribution(data points concentration and spread of values)
    subplot(4, 3, i);
    histogram(data{:,i}); % extract all rows from i(th) column resulting column vector with all the values for that feature, {} returns data in array
    title(data.Properties.VariableNames{i});
end

% Scale and center the dataset
data_scaled = zscore(data{:,1:end-1});
figure;
subplot(1, 1, 1);
boxplot(data_scaled);
title('Boxplots of Scaled Variables');

% Apply PCA
[coeff, score, latent, tsquared, explained] = pca(data_scaled);

% Visualize the variation explained
figure;
subplot(1, 1, 1);
pareto(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Variance Explained by Each Principal Component');

% Compute the biplot, first two principal components
figure;
subplot(1, 1, 1);
biplot(coeff(:,1:2), 'Scores', score(:,1:2), 'VarLabels', data.Properties.VariableNames(1:end-1));
title('Biplot of the First Two Principal Components');

% Quality vs PCs
quality = data.quality;
figure;
subplot(1, 1, 1);
scatter(score(:,1), score(:,2), [], quality, 'filled');
colorbar;
xlabel('PC1');
ylabel('PC2');
title('Wine Quality vs First Two Principal Components');

% Loading bar plots
figure;
subplot(1, 1, 1);
bar(coeff(:,1));
xticklabels(data.Properties.VariableNames(1:end-1));
title('Loadings of Variables on PC1');

% T² and SPE control charts
T2 = tsquared;
SPE = sum(score(:,3:end).^2, 2);
control_limit_T2 = mean(T2) + 3*std(T2);

figure;
subplot(2, 1, 1);
plot(T2);
hold on;
yline(control_limit_T2, 'r--', '3-sigma limit');
title('T² Control Chart');
xlabel('Observation');
ylabel('T² Statistic');

control_limit_SPE = mean(SPE) + 3*std(SPE);
subplot(2, 1, 2);
plot(SPE);
hold on;
yline(control_limit_SPE, 'r--', '3-sigma limit');
title('SPE Control Chart');
xlabel('Observation');
ylabel('Squared Prediction Error');
