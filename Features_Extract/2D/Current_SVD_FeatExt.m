function [SVDFeature] = Current_SVD_FeatExt(MM)

[U2,S2,V2] = svds(MM,6);
S2_vector = diag(S2);
S2_vector_normalized = (diag(S2)-min(diag(S2)))/(max(diag(S2))-min(diag(S2)));%归一化

Mean_S2 = mean(S2_vector_normalized(2:5,1));
Std_S2 = std(S2_vector_normalized(2:5,1));

P_S2 = diag(S2)/sum(diag(S2));
for i = 1:size(P_S2,1)
    H_S2(i,1) = P_S2(i,1)*log2(P_S2(i,1));
end
Entropy_SVD = -sum(H_S2);


SVDFeature = [double(S2_vector_normalized(2,1)),double(S2_vector_normalized(3,1)),...
              double(S2_vector_normalized(4,1)),double(S2_vector_normalized(5,1)),...
              double(Mean_S2),double(Std_S2),double(Entropy_SVD)];



end