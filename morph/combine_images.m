% path = '12_%d.jpg';
% figure;
% for i = 1:9
%     img = imread(sprintf(path,i));
%     subplot(3,3,i),imshow(img);
% end
% saveas(gcf,'12_auto.jpg');
% 
path = 'auto%d.jpg';
figure;
for i = 1:9
    img = imread(sprintf(path,i+1));
    subplot(3,3,i),imshow(img);
end
saveas(gcf,'morph//12_auto_2.jpg');
% path = 'auto%d.jpg';
% figure;
% for i = 1:11
%     img = imread(sprintf(path,i));
%     all(:,:,:,i) = img;
% end
% montage(all,'Size',[1 11]);
% saveas(gcf,'morph//65_auto_mont.jpg');