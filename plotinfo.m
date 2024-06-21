function plotinfo(infos, metricname)
% Plot the information that algorithms output.
%

    labels = {
        'LR', ...
        'fixed-rank', ...
        '\alpha = 1/20', ...
        '\alpha = 1/2', ...
        '\alpha = 5',
    };

    markers = {
        '-s', ...
        '-o', ...
        '-v', ...
        '-^', ...
        '->',
    };

    colors = {
        '#D95319', ...
        '#007ac7', ...
        '#2f409e', ...
        '#7e2f8e', ...
        '#884a74',
    };

    num_problems = size(infos, 2);

    figure

    subplot(1, 2, 1)

    for i = 1:num_problems
        info = infos{i};
        y = [info.(metricname)];
        semilogy([info.iter], y, markers{i}, 'Color', colors{i}, 'LineWidth', 2, 'MarkerSize', 8);
        hold on
    end

    legend(labels);
    xlabel('Iteration #');
    ylabel(metricname);

    subplot(1, 2, 2)

    for i = 1:num_problems
        info = infos{i};
        y = [info.(metricname)];
        semilogy([info.time], y, markers{i}, 'Color', colors{i}, 'LineWidth', 2, 'MarkerSize', 8);
        hold on
    end

    legend(labels);
    xlabel('Computation time (s)');
    ylabel(metricname);

end
