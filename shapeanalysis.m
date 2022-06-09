% Ariana Pineda, CAAM 210, Spring 2022, Shape Analysis of Curves
% shapeanalysis.m
% This script classifies shapes
% Last Modified: April 1, 2022

function shapeanalysis
% this driver calls and runs functions for open and closed curves with
% provided data

%PART 1

x=0:0.02:pi;
Sinx=[x;sin(x)];
Cosx=[x;cos(x)];
X2=[x;2*x];
X=[x;x];

randSin = NoiseGenerator(Sinx,10);
randCos = NoiseGenerator(Cosx,10);
randX2 = NoiseGenerator(X2,10);
randX = NoiseGenerator(X,10);


curves=cat(3,randSin,randCos,randX2,randX);
PWCM=zeros(40,40);
%find distance between curves
for n=1:40
    for m=1:40
        [Dist]=DistanceCalculator(curves(:,:,m),curves(:,:,n));
        PWCM(n,m)=Dist;
    end
end

figure
imagesc(PWCM)
colorbar
title('Pairwise Comparison Matrix')
xlabel('Index of Matrix Column')
ylabel('Index of Matrix Row')
hold off

%test data
randSinPlot = NoiseGenerator(Sinx,25);
randCosPlot = NoiseGenerator(Cosx,25);
randX2Plot = NoiseGenerator(X2,25);
randXPlot = NoiseGenerator(X,25);
testrand = cat(3,randSinPlot,randCosPlot,randX2Plot, randXPlot);
classifyOpenCurves(curves,testrand);

% PART 2
classifyClosedCurves

end

function [nC] = NoiseGenerator(C,num)
% generates noisy curves
%inputs: C—a curve, num: number of noisy curves we want to generate
%outputs: noisy curve matrix
%initializing variables
[m,n] = size(C);
nC = zeros(m,n,num);

%adding noise to curve
for k = 1:num
    for i = 1:m
        for j=1:n
            nC(i,j,k) = C(i,j) + randn()*.05;
        end
    end
end
end


function[pC] = CurveProcessor(C)
%rescales and recenters the curve
%inputs: C—curve matrix
%outputs: pC–processed curve matrix
n = length(C);
[M]=mean(C,2)

%recentering
centerC(1,:) = C(1,:,:)-M(1);
centerC(2,:)=C(2,:,:)-M(2);

pC=centerC/norm(centerC,'fro');

end



function [Distance]=DistanceCalculator(Cv1,Cv2)
%calculates distance between 2 open curves
%inputs: Cv1–curve1, Cv2—curve2
%outputs: Distance—distance between curves

%calculating distance between the curves
A=Cv1*Cv2';
[U,~,V]=svd(A);
if det(U*V')>0
    Distance=norm(Cv1-(U*V')*Cv2,'fro');
else
    Distance=norm(Cv1-(U*[1 0; 0 -1]*V')*Cv2,'fro');
end
end


function classifyOpenCurves(training, test)
% classifies 100 generated open curves and determines percentage of correctly
% classified shapes
% inputs: training—traning data, test—test data
% outputs: none


count = 0;
%calculates distance between shapes
for i = 1:100
    for j = 1:40
        distance(i,j)=DistanceCalculator(test(:,:,i),training(:,:,j));
    end

    % calculates min distance
    [~,I] = min(distance(i,:));

    % determines percentage of correctly classified open curves
    if i<=25 && I <= 10
        count = count+1;
    elseif i<=50 && I <= 20
        count = count+1;
    elseif i<=75 && I <= 30
        count = count+1;
    elseif i<=100 && I <= 40
        count = count+1;
    end
end
disp("The percentage of correctly classified open curves is " + count+ "%")
end

function dist = distClosedShapes(C1, C2)
% calculates distance between 2 closed curves
% inputs: C1–curve 1, C2—curve 2
% outputs: dist—distance

dist = DistanceCalculator(C1, C2);

% iterates through orientation combinations to find shortest distance
% between curves
for i = 1:length(C1)
    newC1 = [C1(:,i+1:end),C1(:,1:i)];
    dist2 = DistanceCalculator(newC1, C2);
    if dist2<dist
        dist = dist2;
    end
end
end

function classifyClosedCurves
% classifies closed curves and returns percentage of correctly classified
% curves
% inputs: none
% outputs: none

% load data
data = load("DataClassification.mat");
trainingData = data.trainingdata;
test = data.testdata;
disp(size(trainingData))
% plot file data
figure
for i = 1:20
    train = trainingData(:,:,15*i);
    hold on
    subplot(4,5,i);
    plot(train(1,:),train(2,:));
    hold off
end

count = 0;
% filling in distance matrix
for i = 1:100
    for j = 1:300
        dist(i,j)=distClosedShapes(test(:,:,i),trainingData(:,:,j));
    end

    % calculating percentage of correctly classified curves

    [~,I] = min(dist(i,:));
    if ceil(i/5) == ceil(I/15)
        count = count+1;
    end
end
disp("The percentage of correctly classified closed curves is " + count + "%")
end


