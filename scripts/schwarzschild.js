function metric(t, r, theta, phi)
{
    return [-1, 1, r * r, r * r * CMath.sin(theta) * CMath.sin(theta)];
}

metric
