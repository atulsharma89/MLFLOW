import mlflow


def calculate_sum(x,y):
    return x*y


if __name__=='__main__':
    #print("running successfully")
    #a=calculate_sum(10,20)
    #print(f"the sum is {a}")
    #starting mlflow server
    with mlflow.start_run():
        x,y=40,89
        a = a=calculate_sum(x,y)

        #tracking the experiment with mlflow
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("a",a)
