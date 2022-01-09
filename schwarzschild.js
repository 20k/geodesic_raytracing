function metric(t, r, theta, phi)
{
    return [-1, 1, r * r, r * r * Math.sin(theta) * Math.sin(theta)];
}

metric
