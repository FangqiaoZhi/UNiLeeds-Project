
import code.model as model

import numpy as np
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as tmf
import pandas as pd

from code.config import validation_round, need_dump

# Get value which is static in one running 
from code.static import lp_method, device, get_net


def train_one_batch(train_loader, optimizer):
    net, _ = get_net()
    running_loss = 0.0
    net.train()
    n = 0
    for i, data in enumerate(train_loader, 0):
        #print('train_one_batch', i)
        optimizer.zero_grad()
        if lp_method:
            inputs_cpu, _, distances_cpu = data
            inputs = inputs_cpu.to(device)
            distances = distances_cpu.to(device)
            
            '''
            outputs = F.softmax(net(inputs), 1)
            outputs = outputs[:,1]
            targets = torch.where(distances>0, 1-0.25/distances, -0.25/distances)
            loss = F.mse_loss(outputs, targets.float())
            '''
            outputs = net(inputs)
            outputs = torch.squeeze(outputs, 1)
            distances = torch.where(distances>0, 1-0.5/distances, -1-0.5/distances)
            loss = F.mse_loss(outputs, distances.float())
        else:
            inputs_cpu, labels_cpu, _ = data
            inputs = inputs_cpu.to(device)
            labels = labels_cpu.to(device)
            #torch.squeeze(labels, 1)
            outputs = F.log_softmax(net(inputs), 1)
            loss = F.nll_loss(outputs, labels.long())
            
        loss.backward()
        optimizer.step()
        # accumulate loss
        running_loss += loss.item()
        n += 1
    return running_loss/n


def predict_method(inputs):
    net, _ = get_net()
    outputs = net(inputs)
    if lp_method:
        '''
        outputs = F.softmax(outputs, 1)
        outputs = outputs[:,1]
        predicted = torch.where(outputs >= 0.5, 1, 0)  
        '''
        outputs = torch.squeeze(outputs, 1)
        predicted = torch.where(outputs > 0, 1, 0)
    else:
        outputs = F.log_softmax(outputs, 1)
        _, predicted = torch.max(outputs, 1)
    return outputs, predicted
    
    
def predict_method_with_threshold(inputs, threshold):
    net, _ = get_net()
    outputs = net(inputs)
    if lp_method:
        '''
        outputs = F.softmax(outputs, 1)
        outputs = outputs[:,1]
        predicted = torch.where(outputs >= 0.5, 1, 0)  
        '''
        outputs = torch.squeeze(outputs, 1)
        predicted = torch.where(outputs > 0, 1, 0)
    else:
        outputs = F.softmax(outputs, 1)
        outputs = outputs.permute(1, 0, 2, 3)
        outputs = outputs[1]
        predicted = torch.where(outputs>threshold, 1, 0)
        predicted = predicted.int()
    return outputs, predicted


def calc_loss(outputs, labels, distances):
    if lp_method:
        #targets = torch.where(distances>0, 1-0.25/distances, -0.25/distances)
        targets = torch.where(distances>0, 1-0.5/distances, -1-0.5/distances)
        #targets = torch.where(distances>0, 2-1/distances, -2-1/distances)
        loss = F.mse_loss(outputs, targets.float()) 
    else:
        labels = labels.long()
        loss = F.nll_loss(outputs, labels)
    return loss


#validation set function to get the loss and accuracy
def test_one_batch(loader, need_wrong_answer=False):
    #correct = 0 total = 0
    running_loss, n = 0, 0
    positive, negative, false_negative, false_positive = 0, 0, 0, 0
    
    wrong_answers = [0 for p in range(2000)]
    all_answers = [0 for p in range(2000)]
    net, _ = get_net()
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs_cpu, labels_cpu, distances_cpu = data
            inputs, labels, distances = inputs_cpu.to(device), labels_cpu.to(device), distances_cpu.to(device)
                
            outputs, predicted = predict_method(inputs)
            loss = calc_loss(outputs, labels, distances)

            predicted = predicted.int()
            labels = labels.int()
            zeros = torch.zeros(labels.shape).to(device).int()
            positive       += torch.where(predicted==labels, labels, zeros).sum().cpu()
            negative       += torch.where(predicted==labels, 1-labels, zeros).sum().cpu()
            false_negative += torch.where(predicted!=labels, labels, zeros).sum().cpu()
            false_positive += torch.where(predicted!=labels, 1-labels, zeros).sum().cpu()

            if need_wrong_answer:     
                distances = distances.int()
                #print('shapes', predicted.shape, labels.shape, distances.shape, zeros.shape)
                wrong_answer = torch.where(predicted!=labels, distances, zeros)
                all_answer = distances
                wrong_answer = wrong_answer.cpu().numpy()
                all_answer = all_answer.cpu().numpy()
                for p in wrong_answer:
                    for q in p:
                        for r in q:
                            wrong_answers[r+1000] = wrong_answers[r+1000] + 1
                for p in all_answer:
                    for q in p:
                        for r in q:
                            all_answers[r+1000] = all_answers[r+1000] + 1

            n += 1
            running_loss += loss.item()

    if need_wrong_answer:
        return running_loss/n, positive, negative, false_negative, false_positive, wrong_answers, all_answers
    else:
        return running_loss/n, positive, negative, false_negative, false_positive
    
    
    
#validation set function to get the loss and accuracy
def test_with_threshold(loader, need_wrong_answer=False, threshold=0.5):
    net, _ = get_net()
    net.eval()
    positive, negative, false_negative, false_positive = 0, 0, 0, 0
    
    wrong_answers = [0 for p in range(2000)]
    all_answers = [0 for p in range(2000)]
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs_cpu, labels_cpu, distances_cpu = data
            inputs, labels, distances = inputs_cpu.to(device), labels_cpu.to(device), distances_cpu.to(device)  
                                
            outputs = net(inputs)
            
            '''
            outputs = F.softmax(outputs, 1)
            outputs = outputs.permute(1, 0, 2, 3)
            outputs = outputs[1]
            labels = labels.int()
            '''
            outputs = torch.squeeze(outputs, 1)
            predicted = torch.where(outputs > 0, 1, 0)
            #outputs = F.softmax(outputs, 1)
            #outputs = outputs.permute(1, 0, 2, 3)
            #outputs = outputs[1]
            labels = labels.int()
            
            #predicted = torch.where(outputs>threshold, 1, 0)
            #predicted = torch.where(outputs>0, 1, 0)
            #_, predicted = torch.max(outputs, 1)
            #labels = images.long()
            predicted = predicted.int()
            zeros = torch.zeros(labels.shape).to(device).int()
            #print('images1', images1)
            not_labels = 1 - labels
            
            positive       += torch.where(predicted==labels, labels, zeros).sum().cpu()
            negative       += torch.where(predicted==labels, 1-labels, zeros).sum().cpu()
            false_negative += torch.where(predicted!=labels, labels, zeros).sum().cpu()
            false_positive += torch.where(predicted!=labels, 1-labels, zeros).sum().cpu()

            if need_wrong_answer:     
                distances = distances.int()
                #print('shapes', predicted.shape, labels.shape, distances.shape, zeros.shape)
                wrong_answer = torch.where(predicted!=labels, distances, zeros)
                all_answer = distances
                wrong_answer = wrong_answer.cpu().numpy()
                all_answer = all_answer.cpu().numpy()
                for p in wrong_answer:
                    for q in p:
                        for r in q:
                            wrong_answers[r+1000] = wrong_answers[r+1000] + 1
                for p in all_answer:
                    for q in p:
                        for r in q:
                            all_answers[r+1000] = all_answers[r+1000] + 1

    if need_wrong_answer:
        return positive, negative, false_negative, false_positive, wrong_answers, all_answers
        #return running_loss/n, positive, negative, false_negative, false_positive, wrong_answers, all_answers
    else:
        return positive, negative, false_negative, false_positive
        #return running_loss/n, positive, negative, false_negative, false_positive


def train(round, train_loader, validate_loader):  
    net, _ = get_net()  
    statsrec = np.zeros((6, (round+validation_round-1)//validation_round)) 
    cmt = torch.zeros(2, 2, dtype=torch.int64)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # loop over the dataset multiple times
    for epoch in range(round):
        print ('epoch', epoch, 'time', time.asctime(time.localtime(time.time())))

        ltrn = train_one_batch(train_loader, optimizer)
        #if epoch % dump_round == dump_round - 1:
        if epoch % validation_round != validation_round - 1:
            continue
        if need_dump:
            torch.save(net, f'models/{epoch+1:05d}.pkl')
        ltst, positive, negative, false_negative, false_positive = test_one_batch(validate_loader)
        #ltst, positive, negative, false_negative, false_positive, was, als = test_one_batch(validate_loader, True)
        #print('was', was, als)
        '''
        for p in range(200):
            if als[1000+p] != 0:
                print(f'+{p},{was[1000+p]}/{als[1000+p]}={100.0*was[1000+p]/als[1000+p]}%')
            if als[1000-p] != 0:
                print(f'-{p},{was[1000-p]}/{als[1000-p]}={100.0*was[1000-p]/als[1000-p]}%')
        '''
        cmt = torch.Tensor([[negative, false_positive],[false_negative, positive],]).int()
        right = positive + negative
        wrong = false_negative + false_positive
        total = right + wrong
        accuracy = right / total
        precision = positive / (positive + false_positive)
        recall = positive / (positive + false_negative)
        f1_score = 2*precision*recall/(precision+recall)
        
        loss_str = f'Epoch:{epoch} TrainLoss:{ltrn: .5f} ValidationLoss: {ltst: .5f}'
        rate_str = f'Accuracy:{accuracy: .5f}, Precision:{precision: .5f}, Recall:{recall: .5f}'
        f1_str = f'F1:{f1_score: .5f}'
        count_str = f'+: {positive}, -: {negative}, False+: {false_negative}, False-: {false_positive}'
        print(f'{loss_str}, {rate_str} {f1_str}, {count_str}')
        statsrec[:,epoch//validation_round] = (ltrn, ltst, accuracy, precision, recall, f1_score) 
    return statsrec, cmt