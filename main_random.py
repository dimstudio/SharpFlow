import json
import select
import socket
import random
import threading

bind_ip = '127.0.0.1'
port_list = [20001]  # could work with 2 ports
servers = []
targets = ['classRelease', 'classDepth', 'classRate', 'armsLocked', 'bodyWeight']
request_count = 0


def start_tcp_server(ip, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    server.listen(5)  # max backlog of connections
    print(('Listening on {}:{}'.format(ip, port)))
    servers.append(server)


def handle_client_connection(client_socket, port):
    return_dict = {}
    for t in targets:
        # uncomment this in case of 3 classes
        #if t == 'classDepth' or t == 'classRate':
        #    return_dict[t] = random.randrange(3)
        #else:
        return_dict[t] =  random.randrange(2)
    print(return_dict)
    client_socket.send(str(return_dict).encode())
    #client_socket.close()


if __name__ == '__main__':

    for port in port_list:
        start_tcp_server(bind_ip, port)

    while True:
        readable, _, _ = select.select(servers, [], [])
        ready_server = readable[0]
        request_count = request_count + 1
        connection, address = ready_server.accept()
        print(('Accepted connection from {}:{}'.format(address[0], address[1])))
        client_handler = threading.Thread(
            target=handle_client_connection,
            args=(connection, port)
        )
        client_handler.start()