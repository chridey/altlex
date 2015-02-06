import sys
import json
import SocketServer

from chnlp.mixture.mixture import ConstrainedMixture

    #need to handle both training and testing
    #for testing need to send back results

class TCPHandler(SocketServer.StreamRequestHandler):

    def handle(self):
        # self.rfile is a file-like object created by the handler;
        # we can now use e.g. readline() instead of raw recv() calls
        data = self.rfile.readline().strip()
        print ("{} wrote:".format(self.client_address[0]))

        #print data
        dataDict = json.loads(data)
        features = dataDict['data']
        if 'class_values' in dataDict:
            labels = dataDict['class_values']
            self.cm = ConstrainedMixture(len(features[0]))
            self.cm.train(features, labels)
            self.wfile.write('finished training')
        else:
            labels = self.cm.classify(features)
            print(len(labels))
            # Likewise, self.wfile is a file-like object used to write back
            # to the client
            ret = json.dumps(labels)
            self.wfile.write(ret + '\n')

if __name__ == "__main__":
    HOST = "localhost"
    if len(sys.argv) > 1:
        PORT = int(sys.argv[1])
    else:
        PORT = 8888

    # Create the server, binding to localhost on port 8888
    server = SocketServer.TCPServer((HOST, PORT), TCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
