clear; clc; close all; 

rpm = [7.18, 1273.35, 586.78, 2596.87];
kc = [7.16, 43.03, 25.70, 32.95];
vbp = [12.02, 140.97, 140.53, 136.26];
gg = [0.07, 1.91, 1.97, 1.76];

% y = log([rpm, kc, vbp, gg]);
y = [rpm; kc; vbp; gg];
y = y(:)

x = [1 : numel(y)]';
% bar(y, 0.2)
bar(x,y, 0.2)

labels = arrayfun(@(value) num2str(value,'%2.1f'),y,'UniformOutput',false);
text(x,y,labels,...
  'HorizontalAlignment','center',...
  'VerticalAlignment','bottom') 