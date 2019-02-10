% path = '12_%d.jpg';
% figure;
% for i = 1:9
%     img = imread(sprintf(path,i));
%     subplot(3,3,i),imshow(img);
% end
% saveas(gcf,'12_auto.jpg');
% 
path = 'manual%d.jpg';
figure;
for i = 1:9
    img = imread(sprintf(path,i+1));
    subplot(3,3,i),imshow(img);
end
saveas(gcf,'34_manual.jpg');