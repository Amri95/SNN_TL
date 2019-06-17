import torch


def linear_kernel(X):
    return torch.mm(X, torch.t(X))


def HSIC(K, L):
    n = K.size()[0]
    H = torch.eye(n).cuda() - 1/n * torch.ones((n, n)).cuda()

    result = torch.mm(torch.mm(K, H), torch.mm(L, H))
    result = 1/((n-1)**2) * torch.trace(result)

    return result


def linear_CKA(X, Y):
    K = linear_kernel(X)
    # K = K + torch.ones(K.size()).cuda() * 1e-3
    L = linear_kernel(Y)
    # L = L + torch.ones(L.size()).cuda() * 1e-3
    # print(HSIC(K, L))
    return HSIC(K, L) / (torch.sqrt(HSIC(K, K) * HSIC(L, L)) + torch.tensor(1e-5).cuda())


def RBF_CKA(X, Y):
    K = linear_kernel(X)
    L = linear_kernel(Y)
    return HSIC(K, L) / torch.sqrt(HSIC(K, K) * HSIC(L, L))