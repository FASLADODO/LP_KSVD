a_sum=zeros(570,1216);
% for i=1:100
%     a_sum=a_sum+Q_cum{i};
% end
for i=1:99
    a_s{i}=Q_cum{i}-Q_cum{i+1};
end
for i=1:99
    a_sum=a_sum+a_s{i};
end
% a_sum=abs(a_sum);
% for i=1:size(a_sum,1)
%     for j=1:size(a_sum,2)
%         if a_sum(i,j)>0.85
%             a_sum(i,j)=1;
%         else
%             a_sum(i,j)=0;
%         end
%     end
% end
imshow(a_sum)