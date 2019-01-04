#include <caffe2/core/init.h>
// #include "../../../include/caffe2/util/blob.h"
// #include "../../../include/caffe2/util/model.h"
// #include "../../../include/caffe2/util/net.h"
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>
#include <caffe2/core/blob.h>
#include <caffe2/core/tensor.h>

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

CAFFE2_DEFINE_string(train_db, "res/mnist-train-nchw-leveldb",
                     "The given path to the training leveldb.");
CAFFE2_DEFINE_string(test_db, "res/mnist-test-nchw-leveldb",
                     "The given path to the testing leveldb.");
CAFFE2_DEFINE_int(iters, 100, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 50, "The of test runs.");
CAFFE2_DEFINE_bool(force_cpu, false, "Only use CPU, no CUDA.");
CAFFE2_DEFINE_bool(display, false, "Display graphical training info.");

namespace caffe2 {
  
                  const std::set<std::string> trainable_ops({
                    "Add",
                    "AffineScale",
                    "AveragedLoss",
                    "AveragePool",
                    "BackMean",
                    "Concat",
                    "Conv",
                    "Diagonal",
                    "Dropout",
                    "EnsureCPUOutput",
                    "FC",
                    "LabelCrossEntropy",
                    "LRN",
                    "MaxPool",
                    "Mul",
                    "RecurrentNetwork",
                    "Relu",
                    "Reshape",
                    "Slice",
                    "Softmax",
                    "SpatialBN",
                    "SquaredL2",
                    "SquaredL2Channel",
                    "StopGradient",
                    "Sum",
                });

                const std::set<std::string> non_trainable_ops({
                    "Accuracy",
                    "Cast",
                    "Cout",
                    "ConstantFill",
                    "Iter",
                    "Scale",
                    "TensorProtosDBInput",
                    "TimePlot",
                    "ShowWorst",
                });

                const std::map<std::string, std::string> custom_gradient({
                    {"EnsureCPUOutput", "CopyFromCPUInput"},
                    {"CopyFromCPUInput", "EnsureCPUOutput"},
                });

                const std::set<std::string> pass_gradient({"Sum"});

                const std::set<std::string> filler_ops({
                    "UniformFill",
                    "UniformIntFill",
                    "UniqueUniformFill",
                    "ConstantFill",
                    "GaussianFill",
                    "XavierFill",
                    "MSRAFill",
                    "RangeFill",
                    "LengthsRangeFill",
                });
                const std::string gradient_suffix("_grad");


      OperatorDef* AddWeightedSumOp(NetDef net, const std::vector<std::string>& inputs,const std::string& sum);
      std::vector<OperatorDef> CollectGradientOps( NetDef& net, std::map<std::string, std::pair<int, int>>& split_inputs) ;
      OperatorDef* AddGradientOps(NetDef& net, OperatorDef& op, std::map<std::string, std::pair<int, int>>& split_inputs,std::map<std::string, 
                                                                              std::string>& pass_replace,std::set<std::string>& stop_inputs) ;
      OperatorDef* AddGradientOp(NetDef& net,OperatorDef& op);
      size_t Write(NetDef& init,NetDef& predict,  std::string &path_prefix);



        // model.predict.AddWeightedSumOp({param, "ONE", param + "_grad", "LR"}, param);
      OperatorDef* AddWeightedSumOp(NetDef net, const std::vector<std::string>& inputs,const std::string& sum) {
        auto op = net.add_op();
        op->set_type("WeightedSum");
        for (auto input : inputs) {
          op->add_input(input);
        }
        for (auto output : {sum}) {
          op->add_output(output);
        }
        return op;
      }


      void  AddGradientOps( NetDef& target){
        std::map<std::string, std::pair<int, int>> split_inputs;//?
        std::map<std::string, std::string> pass_replace;//?
        std::set<std::string> stop_inputs;//?
        auto ops = CollectGradientOps(target, split_inputs);
        for (auto op : ops) {
          AddGradientOps(target, op, split_inputs, pass_replace, stop_inputs);//输入的都为空?
        }
      }


      std::vector<OperatorDef> CollectGradientOps( NetDef& net, std::map<std::string, std::pair<int, int>>& split_inputs) 
      {
          // std::set<std::string> external_inputs(net.external_input().begin(),net.external_input().end());//定义external_inputs
          std::vector<OperatorDef> gradient_ops;
          std::map<std::string, int> input_count;
          for (auto& op : net.op()) {
            if (trainable_ops.find(op.type()) != trainable_ops.end()) {//验证是不会定义的op类型
              gradient_ops.push_back(op);//堆栈可训练的ops
              for (auto& input : op.input()) {
                auto& output = op.output();
                if (std::find(output.begin(), output.end(), input) == output.end()) {//?
                  input_count[input]++;
                  if (input_count[input] > 1) {
                    split_inputs[input + gradient_suffix] = {input_count[input],input_count[input]};
                  }
                }
              }
            } else if (non_trainable_ops.find(op.type()) == non_trainable_ops.end()) {
              CAFFE_THROW("unknown backprop operator type: " + op.type());
            }
          }
          std::reverse(gradient_ops.begin(), gradient_ops.end());
          return gradient_ops;
      }

      bool net_util_op_has_output1(const OperatorDef& op, const std::set<std::string>& names) {
          for (const auto& output : op.output()) {
            if (names.find(output) != names.end()) {
              return true;
            }
          }
          return false;
      }


      OperatorDef* AddGradientOps(NetDef& net, OperatorDef& op, std::map<std::string, std::pair<int, int>>& split_inputs,std::map<std::string, std::string>& pass_replace,
          std::set<std::string>& stop_inputs) {
          OperatorDef* grad = NULL;
          if (custom_gradient.find(op.type()) != custom_gradient.end()) {
            grad = net.add_op();
            grad->set_type(custom_gradient.at(op.type()));
            for (auto arg : op.arg()) {
              auto copy = grad->add_arg();
              copy->CopyFrom(arg);
            }
            for (auto output : op.output()) {
              grad->add_input(output + gradient_suffix);
            }
            for (auto input : op.input()) {
              grad->add_output(input + gradient_suffix);
            }
          } else if (pass_gradient.find(op.type()) != pass_gradient.end()) {
            for (auto input : op.input()) {
              auto in = input + gradient_suffix;
              if (split_inputs.count(in) && split_inputs[in].first > 0) {
                split_inputs[in].first--;
                in += "_sum_" + std::to_string(split_inputs[in].first);
              }
              pass_replace[in] = op.output(0) + gradient_suffix;
            }
          } else if (op.type() == "StopGradient" ||
                    net_util_op_has_output1(op, stop_inputs)) {
            for (const auto& input : op.input()) {
              stop_inputs.insert(input);
            }
          } else {
            grad = AddGradientOp(net,op);
            if (grad == NULL) {
              std::cerr << "no gradient for operator " << op.type() << std::endl;
            }
          }
          if (grad != NULL) {
            grad->set_is_gradient_op(true);
            for (auto i = 0; i < grad->output_size(); i++) {
              auto output = grad->output(i);
              if (split_inputs.count(output) && split_inputs[output].first > 0) {
                split_inputs[output].first--;
                grad->set_output(
                    i, output + "_sum_" + std::to_string(split_inputs[output].first));
              }
            }
            for (auto i = 0; i < grad->input_size(); i++) {
              auto input = grad->input(i);
              if (pass_replace.count(input)) {
                grad->set_input(i, pass_replace[input]);
                pass_replace.erase(input);
              }
            }
            // fix for non-in-place SpatialBN
            if (grad->type() == "SpatialBNGradient" &&
                grad->input(2) == grad->output(0)) {
              pass_replace[grad->output(0)] = grad->output(0) + "_fix";
              grad->set_output(0, grad->output(0) + "_fix");
            }
          }
          // merge split gradients with sum
          for (auto& p : split_inputs) {
            if (p.second.first == 0) {
              std::vector<std::string> inputs;
              for (int i = 0; i < p.second.second; i++) {
                auto input = p.first + "_sum_" + std::to_string(i);
                if (pass_replace.count(input)) {
                  auto in = pass_replace[input];
                  pass_replace.erase(input);
                  input = in;
                }
                inputs.push_back(input);
              }

              // AddSumOp(inputs, p.first);
                auto op = net.add_op();
                op->set_type("Sum");
                for (auto input : inputs) {
                  op->add_input(input);
                }
                for (auto output : {p.first}) {
                  op->add_output(output);
                }

              p.second.first--;
            }
          }
          return grad;
      }


      OperatorDef* AddGradientOp(NetDef& net,OperatorDef& op) {
          OperatorDef* grad = NULL;
          vector<GradientWrapper> output(op.output_size());
          for (auto i = 0; i < output.size(); i++) {
            output[i].dense_ = op.output(i) + gradient_suffix;
          }
          GradientOpsMeta meta = GetGradientForOp(op, output);
          if (meta.ops_.size()) {
            for (auto& m : meta.ops_) {
              auto op = net.add_op();
              op->CopyFrom(m);
              if (grad == NULL) {
                grad = op;
              }
            }
          }
          return grad;
      }


      size_t Write_init_net( NetDef& net, const std::string& path) {
        WriteProtoToBinaryFile(net, path+"_init_net.pb");
        return std::ifstream(path, std::ifstream::ate | std::ifstream::binary)
            .tellg();
      }
      size_t Write_predict_net( NetDef& net, const std::string& path) {
        WriteProtoToBinaryFile(net, path+"_predict_net.pb");
        return std::ifstream(path, std::ifstream::ate | std::ifstream::binary)
            .tellg();
      }

      size_t WriteText(NetDef& net,const std::string& path) {
        WriteProtoToTextFile(net, path+"predict_net.pbtx");
        return std::ifstream(path, std::ifstream::ate | std::ifstream::binary)
            .tellg();
      }


  void run() {
    std::cout << std::endl;
    std::cout << "## Caffe2 MNIST Tutorial ##" << std::endl;
    std::cout << "https://caffe2.ai/docs/tutorial-MNIST.html" << std::endl;
    std::cout << std::endl;

    if (!std::ifstream(FLAGS_train_db).good() ||
        !std::ifstream(FLAGS_test_db).good()) {
      std::cerr << "error: MNIST database missing: "
                << (std::ifstream(FLAGS_train_db).good() ? FLAGS_test_db
                                                        : FLAGS_train_db)
                << std::endl;
      std::cerr << "Make sure to first run ./script/download_resource.sh"
                << std::endl;
      return;
    }

    std::cout << "train-db: " << FLAGS_train_db << std::endl;
    std::cout << "test-db: " << FLAGS_test_db << std::endl;
    std::cout << "iters: " << FLAGS_iters << std::endl;
    std::cout << "test-runs: " << FLAGS_test_runs << std::endl;
    std::cout << "force-cpu: " << (FLAGS_force_cpu ? "true" : "false")
              << std::endl;
    std::cout << "display: " << (FLAGS_display ? "true" : "false") << std::endl;

  #ifdef WITH_CUDA
    if (!FLAGS_force_cpu) {
      DeviceOption option;
      option.set_device_type(CUDA);
      new CUDAContext(option);
      std::cout << std::endl << "using CUDA" << std::endl;
    }
  #endif


    // >>> from caffe2.python import core, cnn, net_drawer, workspace, visualize,
    // brew
    // >>> workspace.ResetWorkspace(root_folder)
    Workspace workspace("tmp");

    // >>> train_model = model_helper.ModelHelper(name="mnist_train",
    // arg_scope={"order": "NCHW"})
    NetDef train_init_model, train_predict_model;
    // ModelUtil train(train_init_model, train_predict_model, "mnist_train");

    train_init_model.set_name("mnist_train");
    train_predict_model.set_name("mnist_train");


      // >>> data, label = AddInput(train_model, batch_size=64,
      // db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'),
      // db_type='leveldb')

  /************/   
  //  AddInput(train, 64, FLAGS_train_db, "leveldb");

  {
        //   model.init.AddCreateDbOp("dbreader", db_type, db);
        {
                    auto op = train_init_model.add_op();
                    op->set_type("CreateDB");
                    op->add_output("dbreader");
                    auto arg1 = op->add_arg();
                    arg1->set_name("db_type");        
                    arg1->set_s("leveldb");
                    auto arg2 = op->add_arg();
                    arg2->set_name("db");        
                    arg2->set_s(FLAGS_train_db);
        }
        //  model.predict.AddInput("dbreader"); 
                    train_predict_model.add_external_input("dbreader"); 

        //  model.predict.AddTensorProtosDbInputOp("dbreader", "data_uint8", "label",batch_size) 
        {
                      auto op = train_predict_model.add_op();// model.init.AddCreateDbOp("dbreader", db_type, db);    //   auto op = AddOp("TensorProtosDBInput", {reader}, {data, label});
                      op->set_type("TensorProtosDBInput");
                      op->add_input("dbreader");
                      for (auto output : {"data_uint8", "label"}) {
                        op->add_output(output);
                      }
                      
                      auto arg = op->add_arg();  //   net_add_arg(*op, "batch_size", batch_size);
                      arg->set_name("batch_size");        
                      arg->set_i(64);
        } 
  

        {// model.predict.AddCastOp("data_uint8", "data", TensorProto_DataType_FLOAT);
                      auto op = train_predict_model.add_op();// 
                      op->set_type("Cast");
                      op->add_input("data_uint8");
                      op->add_output("data");
                      auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                      arg->set_name("to");        
                      arg->set_i(TensorProto_DataType_FLOAT);
        }



        //   /* model.predict.AddScaleOp("data", "data", 1.f / 256); */
        {
                      auto op = train_predict_model.add_op();// 
                      op->set_type("Scale");
                      op->add_input("data");
                      op->add_output("data");
                      auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                      arg->set_name("scale");        
                      arg->set_f(1.f / 256);
        }          


        // /* model.predict.AddStopGradientOp("data") */
        {
                    auto op = train_predict_model.add_op();
                    op->set_type("StopGradient");
                    op->add_input("data");
                    op->add_output("data");
        }
    
  }


  /************/   
    // AddLeNetModel(train, false);//false
      
  {      // model.AddConvOps("data", "conv1", 1, 20, 1, 0, 5, test);
        {        
                  //  init.AddXavierFillOp({out_size, in_size, kernel, kernel}, output + "_w"); 
                  {
                    auto op = train_init_model.add_op();// 
                    op->set_type("XavierFill");
                    op->add_output({"conv1_w"});
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("shape");
                    for (auto value :{20, 1, 5, 5}) {
                        arg->add_ints(value);
                      }        
                  }
                    
                  {//     init.AddConstantFillOp({out_size}, output + "_b");
                    auto op = train_init_model.add_op();// 
                    op->set_type("ConstantFill");
                    op->add_output("conv1_b");
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("shape"); 
                    for (auto value : {20}) {
                        arg->add_ints(value);
                      }       
                  }
                              
                    
                  train_predict_model.add_external_input("conv1_w");// predict.AddInput(output + "_w");
                  train_predict_model.add_external_input("conv1_b");//predict.AddInput(output + "_b");


                  //   predict.AddConvOp(input, output + "_w", output + "_b", output, stride, padding, kernel)
                  // OperatorDef* NetUtil::AddConvOp(const std::string& input, const std::string& w, const std::string& b, const std::string& output,int stride, int padding, int kernel, int group, const std::string& order) 
                  { 
              
                    auto op = train_predict_model.add_op();// 
                    op->set_type("Conv");
                    // "conv1_b".size() ? std::vector<std::string>({"data", "conv1_w", "conv1_b"}): std::vector<std::string>({"data",, "conv1_w"});
                    for (auto input :{"data", "conv1_w","conv1_b"}) {//{"data", "conv1_w", "conv1_b"}
                        op->add_input(input);
                      } 
                    op->add_output("conv1");
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("stride");        
                    arg->set_i(1);
                    auto arg1 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg2->set_name("kernel");        
                    arg2->set_i(5);
                    // if (group != 0) 
                    // auto arg3 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    // arg->set_name("group");        
                    // arg->set_i(0);
                    auto arg4 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg4->set_name("order");        
                    arg4->set_s("NCHW");
                  }
        }
    
        // model.predict.AddMaxPoolOp("conv1", "pool1", 2, 0, 2);
        {      

                    auto op = train_predict_model.add_op();// 
                    op->set_type("MaxPool");
                    op->add_input("conv1");
                    op->add_output("pool1");
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("stride");        
                    arg->set_i(2);
                    auto arg1 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg2->set_name("kernel");        
                    arg2->set_i(2);
                    // if (group != 0) 
                    auto arg3 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg3->set_name("order");        
                    arg3->set_s("NCHW");
                    auto arg4 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg4->set_name("legacy_pad");        
                    arg4->set_i(3);
        }



        // void ModelUtil::AddConvOps(const std::string &input, const std::string &output,int in_size, int out_size, int stride, int padding,int kernel, bool test)
        //   model.AddConvOps("pool1", "conv2", 20, 50, 1, 0, 5, test);   

        {      
                  {//  init.AddXavierFillOp({out_size, in_size, kernel, kernel}, output + "_w"); 
                    auto op = train_init_model.add_op();// 
                    op->set_type("XavierFill");
                    op->add_output({"conv2_w"});
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("shape");
                    for (auto value :{50, 20, 5, 5}) {
                        arg->add_ints(value);
                      }        
                  }
                    
                  {//     init.AddConstantFillOp({out_size}, output + "_b");
                    auto op = train_init_model.add_op();// 
                    op->set_type("ConstantFill");
                    op->add_output("conv2_b");
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("shape");        
                    for (auto value :{50}) {
                        arg->add_ints(value);
                      } 
                  }
                  // if (!test) 启动以上两段      

                    
                  train_predict_model.add_external_input("conv2_w");// predict.AddInput(output + "_w");
                  train_predict_model.add_external_input("conv2_b");//predict.AddInput(output + "_b");
                  //   predict.AddConvOp(input, output + "_w", output + "_b", output, stride, padding, kernel)
                
                  { 
              
                    auto op = train_predict_model.add_op();// 
                    op->set_type("Conv");
                    // op->add_input("conv1_b".size() ? std::vector<std::string>({"data", "conv1_w", "conv1_b"}) : std::vector<std::string>({"data", "conv1_w"}));
                  //  auto bbbb= workspace.GetBlob("conv2_b")->size();///???
                    for (auto input :{"pool1", "conv2_w", "conv2_b"}) {
                        op->add_input(input);
                      } 
                     op->add_output("conv2");
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("stride");        
                    arg->set_i(1);
                    auto arg1 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg2->set_name("kernel");        
                    arg2->set_i(5);
                    // if (group != 0) 
                    // auto arg3 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    // arg->set_name("group");        
                    // arg->set_i(0);
                    auto arg3 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg3->set_name("order");        
                    arg3->set_s("NCHW");
                  }
        }


        // model.predict.AddMaxPoolOp("conv2", "pool2", 2, 0, 2);
        {      

                    auto op = train_predict_model.add_op();// 
                    op->set_type("MaxPool");                 
                    op->add_input("conv2");
                    op->add_output("pool2");

                    auto arg = op->add_arg();  
                    arg->set_name("stride");        
                    arg->set_i(2);
                    auto arg1 = op->add_arg();  
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg(); 
                    arg2->set_name("kernel");        
                    arg2->set_i(2);
                    auto arg3 = op->add_arg(); 
                    arg3->set_name("order");        
                    arg3->set_s("NCHW");
                    auto arg4 = op->add_arg();  
                    arg4->set_name("legacy_pad");        
                    arg4->set_i(3);
        }


      // model.AddFcOps("pool2", "fc3", 800, 500, test);
      {      
                  {//  init.AddXavierFillOp({out_size, in_size}, output + "_w");; 
                    auto op = train_init_model.add_op();// 
                    op->set_type("XavierFill");
                    op->add_output({"fc3_w"});
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("shape");
                    for (auto value :{500, 800}) {
                        arg->add_ints(value);
                      }        
                  }

                  {//     init.AddConstantFillOp({out_size}, output + "_b");
                    auto op = train_init_model.add_op();// 
                    op->set_type("ConstantFill");
                    op->add_output("fc3_b");
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("shape");        
                    for (auto value :{500}) {
                        arg->add_ints(value);
                      }  

                  }
                  // if (!test) 启动以上两段                
                  train_predict_model.add_external_input("fc3_w");// predict.AddInput(output + "_w");
                  train_predict_model.add_external_input("fc3_b");//predict.AddInput(output + "_b");

                    // auto op = AddOp("FC", {input, w, b}, {output});
                  { 
              
                    auto op = train_predict_model.add_op();// 
                    op->set_type("FC");
                    for (auto input : {"pool2", "fc3_w", "fc3_b"}) {
                      op->add_input(input);
                    }
                    op->add_output("fc3");

                    // // if (axis != 1)
                    // auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    // arg->set_name("axis");        
                    // arg->set_i(1);
                    
                  }
        }

        //   // >>> fc3 = brew.relu(model, fc3, fc3)
        //   model.predict.AddReluOp("fc3", "fc3");
        {
                    auto op = train_predict_model.add_op();// 
                    op->set_type("Relu");
                    op->add_input("fc3");
                    op->add_output("fc3");

        }
        //   model.AddFcOps("fc3", "pred", 500, 10, test); 
        {       
                  {//  init.AddXavierFillOp({out_size, in_size}, output + "_w");; 
                    auto op = train_init_model.add_op();// 
                    op->set_type("XavierFill");
                    op->add_output({"pred_w"});
                    auto arg = op->add_arg(); 
                    arg->set_name("shape");
                    for (auto value :{10, 500}) {
                        arg->add_ints(value);
                      }        
                  }
                    
                  {//     init.AddConstantFillOp({out_size}, output + "_b");
                    auto op = train_init_model.add_op();// 
                    op->set_type("ConstantFill");
                    op->add_output("pred_b");
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("shape");  
                    for (auto value :{10}) {
                        arg->add_ints(value);
                      }                         
                    // arg->set_i(10);

                  }
                  // if (!test) 启动以上两段             
                    
                  train_predict_model.add_external_input("pred_w");// predict.AddInput(output + "_w");
                  train_predict_model.add_external_input("pred_b");//predict.AddInput(output + "_b");

                  //   predict.AddFcOp(input, output + "_w", output + "_b", output);
                  // OperatorDef* NetUtil::AddFcOp(const std::string& input, const std::string& w,  const std::string& b, const std::string& output, int axis)
                  { 
              
                    auto op = train_predict_model.add_op();// 
                    op->set_type("FC");
                    for (auto input : {"fc3", "pred_w", "pred_b"}) {
                      op->add_input(input);
                    }
                    op->add_output("pred");

                  // // if (axis != 1)
                    // auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    // arg->set_name("axis");        
                    // arg->set_i(1);          
                  }
        }

          // model.predict.AddSoftmaxOp("pred", "softmax");
      
        {          
                    auto op = train_predict_model.add_op();// 
                    op->set_type("Softmax");
                    op->add_input("pred");
                    op->add_output("softmax");
                    // auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    // arg->set_name("axis");        
                    // arg->set_i(1);                  
        }
  }


    // /* model.predict.AddStopGradientOp("data") */
    // >>> AddTrainingOperators(train_model, softmax, label)

    /************/   
    // AddTrainingOperators(train);
  {
        
        // model.predict.AddLabelCrossEntropyOp("softmax", "label", "xent");
        {          
                    auto op = train_predict_model.add_op();// 
                    op->set_type("LabelCrossEntropy");
                    for (auto input : {"softmax", "label"}) {
                      op->add_input(input);
                    }
                    op->add_output("xent");                 
        }
        //  model.predict.AddAveragedLossOp("xent", "loss");
        {          
                    auto op = train_predict_model.add_op();// 
                    op->set_type("AveragedLoss");
                    op->add_input("xent");
                    op->add_output("loss");
                
        }


        //  AddAccuracy(model);
        {          
                    // model.predict.AddAccuracyOp("softmax", "label", "accuracy");
                    {
                    auto op = train_predict_model.add_op();// 
                    op->set_type("Accuracy");
                    for (auto input : {"softmax", "label"}) {
                      op->add_input(input);
                    }
                    op->add_output("accuracy");

                    }

                    // model.AddIterOps();
                    {
                          {//     init.AddConstantFillOp({out_size}, output + "_b");  AddConstantFillOp({1}, (int64_t)0, iter_name)
                          auto op = train_init_model.add_op();// 
                          op->set_type("ConstantFill");
                          op->add_output("iter");
                          auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                          arg->set_name("shape"); 
                          for (auto value : {1}) {
                              arg->add_ints(value);
                            }  
                          auto arg1 = op->add_arg();
                          arg1->set_name("value");        
                          arg1->set_i(0);    
                          auto arg2 = op->add_arg();
                          arg2->set_name("dtype");        
                          arg2->set_i(TensorProto_DataType_INT64);  
                          }
                          train_init_model.mutable_device_option()->set_device_type(CPU);

                          // predict.AddInput(iter_name);
                          train_predict_model.add_external_input("iter");

                          // predict.AddIterOp(iter_na;
                          {
                          auto op = train_predict_model.add_op();// 
                          op->set_type("Iter");
                          op->add_input("iter");
                          op->add_output("iter");
                          }
                    }
        }     

        // model.predict.AddConstantFillWithOp(1.0, "loss", "loss_grad");
        {
                    auto op = train_predict_model.add_op();// 
                    op->set_type("ConstantFill");
                    op->add_input("loss");
                    op->add_output("loss_grad");
                    auto arg2 = op->add_arg();
                    arg2->set_name("value");        
                    arg2->set_f(1.0);
        }
            
         AddGradientOps( train_predict_model);
        
  }
    // AddTrainingOperators1(train);     
       
        //  model.predict.AddLearningRateOp("iter", "LR", 0.1);
        {
                    auto op = train_predict_model.add_op();// 
                    op->set_type("LearningRate");
                    op->add_input("iter");
                    op->add_output("LR");
                    auto arg = op->add_arg(); 
                    arg->set_name("policy");      
                    arg->set_s("step");  
                    auto arg1 = op->add_arg();
                    arg1->set_name("stepsize");        
                    arg1->set_i(1);    
                    auto arg2= op->add_arg();
                    arg2->set_name("base_lr"); 
                    arg2->set_f(-0.1);
                    auto arg3 = op->add_arg();
                    arg3->set_name("gamma");        
                    arg3->set_f(0.999f);    
      
        }
        // model.init.AddConstantFillOp({1}, 1.f, "ONE");
        {
                    auto op = train_init_model.add_op();// 
                    op->set_type("ConstantFill");
                    op->add_output("ONE");
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("shape"); 
                    for (auto value : {1}) {
                        arg->add_ints(value);
                      }  
                    auto arg2 = op->add_arg();
                    arg2->set_name("value");        
                    arg2->set_f(1.f);    
        }
                    //  model.predict.AddInput("ONE");
                    train_predict_model.add_external_input("ONE");

      {
        std::vector<std::string> params;
            std::set<std::string> external_inputs(train_predict_model.external_input().begin(), train_predict_model.external_input().end());
            for (const auto& op : train_predict_model.op()) {
              auto& output = op.output();
              if (trainable_ops.find(op.type()) != trainable_ops.end()) {
                for (const auto& input : op.input()) {
                  if (external_inputs.find(input) != external_inputs.end()) {
                    if (std::find(output.begin(), output.end(), input) == output.end()) {
                      params.push_back(input);
                    }
                  }
                }
              }
            }
          for (auto param : params )
          {
            AddWeightedSumOp(train_predict_model,{param, "ONE", param + "_grad", "LR"}, param);
          }
            
      }




    /************/   
    // AddBookkeepingOperators(train);

  {        //model.predict.AddPrintOp("accuracy", true);
          {
            auto op = train_predict_model.add_op();// 
            op->set_type("Print");
            op->add_input("accuracy");
            auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);;  
            arg->set_name("to_file");       
            arg->set_i(1);    
          }

          {// model.predict.AddPrintOp("loss", true);
            auto op = train_predict_model.add_op();// 
            op->set_type("Print");
            op->add_input("loss");
            auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);;  
            arg->set_name("to_file");       
            arg->set_i(1);    
          }

          //  for (auto param : model.Params()) 
          // predict.CollectParams()
            std::vector<std::string> params;
            std::set<std::string> external_inputs(train_predict_model.external_input().begin(), train_predict_model.external_input().end());
            for (const auto& op : train_predict_model.op()) {
              auto& output = op.output();
              if (trainable_ops.find(op.type()) != trainable_ops.end()) {
                for (const auto& input : op.input()) {
                  if (external_inputs.find(input) != external_inputs.end()) {
                    if (std::find(output.begin(), output.end(), input) == output.end()) {
                      params.push_back(input);
                    }
                  }
                }
              }
            }
          for (auto param : params )
          {
                  {//model.predict.AddSummarizeOp(param, true);
                    auto op = train_predict_model.add_op();// 
                    op->set_type("Summarize");
                    // op->add_input(param);
                    for (auto input : {param}) {
                        op->add_input(input);
                      }
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);;  
                    arg->set_name("to_file");       
                    arg->set_i(1); 
                  }
                  // 
                  {
                    auto op = train_predict_model.add_op();// 
                    op->set_type("Summarize");
                    // op->add_input(param);
                    for (auto input : {param + "_grad"}) {
                        op->add_input(input);
                      }
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);;  
                    arg->set_name("to_file");       
                    arg->set_i(1);

                  }

          }

  }

    // >>> test_model = model_helper.ModelHelper(name="mnist_test",
    // arg_scope=arg_scope, init_params=False)
    NetDef test_init_model, test_predict_model;
    // ModelUtil test(test_init_model, test_predict_model, "mnist_test");

    test_init_model.set_name("mnist_test");
    test_predict_model.set_name("mnist_test");
    

    // >>> data, label = AddInput(test_model, batch_size=100,
    // db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'), db_type='leveldb')
    // AddInput(test, 100, FLAGS_test_db, "leveldb");

          //   /* model.init.AddCreateDbOp("dbreader", db_type, db);*/
            {
                        auto op = test_init_model.add_op();
                        op->set_type("CreateDB");
                        op->add_output("dbreader");
                        auto arg1 = op->add_arg();
                        arg1->set_name("db_type");        
                        arg1->set_s("leveldb");
                        auto arg2 = op->add_arg();
                        arg2->set_name("db");        
                        arg2->set_s(FLAGS_test_db);
            }
          // //   /* model.predict.AddInput("dbreader"); */
          // //   {
                      test_predict_model.add_external_input("dbreader"); 
          // //   }

          //   /* model.predict.AddTensorProtosDbInputOp("dbreader", "data_uint8", "label",batch_size) */
          {
                        auto op = test_predict_model.add_op();// model.init.AddCreateDbOp("dbreader", db_type, db);    //   auto op = AddOp("TensorProtosDBInput", {reader}, {data, label});
                        op->set_type("TensorProtosDBInput");

                        op->add_input("dbreader");
                        for (auto output : {"data_uint8", "label"}) {
                          op->add_output(output);
                        }
                        auto arg = op->add_arg();  //   net_add_arg(*op, "batch_size", batch_size);
                        arg->set_name("batch_size");        
                        arg->set_i(100);
          } 
          

          {
                        auto op = test_predict_model.add_op();// 
                        op->set_type("Cast");
                        op->add_input("data_uint8");
                        op->add_output("data");
                        auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                        arg->set_name("to");        
                        arg->set_i(TensorProto_DataType_FLOAT);
          }



          //   /* model.predict.AddScaleOp("data", "data", 1.f / 256); */
          {
                        auto op = test_predict_model.add_op();// 
                        op->set_type("Scale");
                        op->add_input("data");
                        op->add_output("data");
                        auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                        arg->set_name("scale");        
                        arg->set_f(1.f / 256);
          }          


            // /* model.predict.AddStopGradientOp("data") */
            {
                        auto op = test_predict_model.add_op();
                        op->set_type("StopGradient");
                        op->add_input("data");
                        op->add_output("data");
            }
            


    // >>> softmax = AddLeNetModel(test_model, data)
    // AddLeNetModel(test, true);

      
  {      // model.AddConvOps("data", "conv1", 1, 20, 1, 0, 5, test);
        {                                  
                    
                  test_predict_model.add_external_input("conv1_w");// predict.AddInput(output + "_w");
                  test_predict_model.add_external_input("conv1_b");//predict.AddInput(output + "_b");


                  //   predict.AddConvOp(input, output + "_w", output + "_b", output, stride, padding, kernel)
                  // OperatorDef* NetUtil::AddConvOp(const std::string& input, const std::string& w, const std::string& b, const std::string& output,int stride, int padding, int kernel, int group, const std::string& order) 
                  { 
              
                    auto op = test_predict_model.add_op();// 
                    op->set_type("Conv");
                    // "conv1_b".size() ? std::vector<std::string>({"data", "conv1_w", "conv1_b"}): std::vector<std::string>({"data",, "conv1_w"});
                    for (auto input :{"data", "conv1_w","conv1_b"}) {//{"data", "conv1_w", "conv1_b"}
                        op->add_input(input);
                      } 
                    op->add_output("conv1");

                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("stride");        
                    arg->set_i(1);
                    auto arg1 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg2->set_name("kernel");        
                    arg2->set_i(5);
                    auto arg4 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg4->set_name("order");        
                    arg4->set_s("NCHW");
                  }
        }
    
        // model.predict.AddMaxPoolOp("conv1", "pool1", 2, 0, 2);
        {      

                    auto op = test_predict_model.add_op();// 
                    op->set_type("MaxPool");
                    op->add_input("conv1");
                    op->add_output("pool1");

                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("stride");        
                    arg->set_i(2);
                    auto arg1 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg2->set_name("kernel");        
                    arg2->set_i(2);
                    // if (group != 0) 
                    auto arg3 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg3->set_name("order");        
                    arg3->set_s("NCHW");
                    auto arg4 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg4->set_name("legacy_pad");        
                    arg4->set_i(3);
        }



        // void ModelUtil::AddConvOps(const std::string &input, const std::string &output,int in_size, int out_size, int stride, int padding,int kernel, bool test)
        //   model.AddConvOps("pool1", "conv2", 20, 50, 1, 0, 5, test);   

        {              
                 test_predict_model.add_external_input("conv2_w");// predict.AddInput(output + "_w");
                  test_predict_model.add_external_input("conv2_b");//predict.AddInput(output + "_b");
                  //   predict.AddConvOp(input, output + "_w", output + "_b", output, stride, padding, kernel)
                
                  { 
              
                    auto op = test_predict_model.add_op();// 
                    op->set_type("Conv");
                    // op->add_input("conv1_b".size() ? std::vector<std::string>({"data", "conv1_w", "conv1_b"}) : std::vector<std::string>({"data", "conv1_w"}));
                    for (auto input :{"pool1", "conv2_w", "conv2_b"}) {
                        op->add_input(input);
                      } 
                      op->add_output("conv2");
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("stride");        
                    arg->set_i(1);
                    auto arg1 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg2->set_name("kernel");        
                    arg2->set_i(5);
                    auto arg4 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg4->set_name("order");        
                    arg4->set_s("NCHW");
                  }
        }


        // model.predict.AddMaxPoolOp("conv2", "pool2", 2, 0, 2);
        {      

                    auto op = test_predict_model.add_op();// 
                    op->set_type("MaxPool");                 
                    op->add_input("conv2");
                    op->add_output("pool2");

                    auto arg = op->add_arg();  
                    arg->set_name("stride");        
                    arg->set_i(2);
                    auto arg1 = op->add_arg();  
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg(); 
                    arg2->set_name("kernel");        
                    arg2->set_i(2);
                    auto arg3 = op->add_arg(); 
                    arg3->set_name("order");        
                    arg3->set_s("NCHW");
                    auto arg4 = op->add_arg();  
                    arg4->set_name("legacy_pad");        
                    arg4->set_i(3);
        }


      // model.AddFcOps("pool2", "fc3", 800, 500, test);
      {      
                              
                  test_predict_model.add_external_input("fc3_w");// predict.AddInput(output + "_w");
                  test_predict_model.add_external_input("fc3_b");//predict.AddInput(output + "_b");

                    // auto op = AddOp("FC", {input, w, b}, {output});
                  { 
              
                    auto op = test_predict_model.add_op();// 
                    op->set_type("FC");
                    for (auto input : {"pool2", "fc3_w", "fc3_b"}) {
                      op->add_input(input);
                    }
                    op->add_output("fc3");
                  }
        }

        //   // >>> fc3 = brew.relu(model, fc3, fc3)
        //   model.predict.AddReluOp("fc3", "fc3");
        {
                    auto op = test_predict_model.add_op();// 
                    op->set_type("Relu");
                    op->add_input("fc3");
                    op->add_output("fc3");

        }
        //   model.AddFcOps("fc3", "pred", 500, 10, test); 
        {       
                 
                  test_predict_model.add_external_input("pred_w");// predict.AddInput(output + "_w");
                  test_predict_model.add_external_input("pred_b");//predict.AddInput(output + "_b");

                  //   predict.AddFcOp(input, output + "_w", output + "_b", output);
                  // OperatorDef* NetUtil::AddFcOp(const std::string& input, const std::string& w,  const std::string& b, const std::string& output, int axis)
                  { 
              
                    auto op = test_predict_model.add_op();// 
                    op->set_type("FC");
                    for (auto input : {"fc3", "pred_w", "pred_b"}) {
                      op->add_input(input);
                    }
                    op->add_output("pred");     
                  }
        }

          // model.predict.AddSoftmaxOp("pred", "softmax");
      
        {          
                    auto op = test_predict_model.add_op();// 
                    op->set_type("Softmax");
                    op->add_input("pred");
                    op->add_output("softmax");                  
        }
  }
    // >>> AddAccuracy(test_model, softmax, label)
    // AddAccuracy(test);

    {          
                    // model.predict.AddAccuracyOp("softmax", "label", "accuracy");
                    {
                    auto op = test_predict_model.add_op();// 
                    op->set_type("Accuracy");
                    for (auto input : {"softmax", "label"}) {
                      op->add_input(input);
                    }
                    op->add_output("accuracy");

                    }

                    // model.AddIterOps();
                    {
                          {//     init.AddConstantFillOp({out_size}, output + "_b");  AddConstantFillOp({1}, (int64_t)0, iter_name)
                          auto op = test_init_model.add_op();// 
                          op->set_type("ConstantFill");
                          op->add_output("iter");
                          auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                          arg->set_name("shape"); 
                          for (auto value : {1}) {
                              arg->add_ints(value);
                            }  
                          auto arg1 = op->add_arg();
                          arg1->set_name("value");        
                          arg1->set_i(0);    
                          auto arg2 = op->add_arg();
                          arg2->set_name("dtype");        
                          arg2->set_i(TensorProto_DataType_INT64);  
                          }
                          test_init_model.mutable_device_option()->set_device_type(CPU);

                          // predict.AddInput(iter_name);
                          test_predict_model.add_external_input("iter");

                          // predict.AddIterOp(iter_na;
                          {
                          auto op = test_predict_model.add_op();// 
                          op->set_type("Iter");
                          op->add_input("iter");
                          op->add_output("iter");
                          }
                    }
        }     





    // >>> deploy_model = model_helper.ModelHelper(name="mnist_deploy",
    // arg_scope=arg_scope, init_params=False)
    NetDef deploy_init_model, deploy_predict_model;
    // ModelUtil deploy(deploy_init_model, deploy_predict_model, "mnist_model");
    deploy_init_model.set_name("mnist_model");
    deploy_predict_model.set_name("mnist_model");

    // deploy.predict.AddInput("data");
    // deploy.predict.AddOutput("softmax");

    deploy_predict_model.add_external_input("data"); 
    deploy_predict_model.add_external_output("softmax");

    // >>> AddLeNetModel(deploy_model, "data")
    // AddLeNetModel(deploy, true);
    {      // model.AddConvOps("data", "conv1", 1, 20, 1, 0, 5, test);
        {                                  
                    
                  deploy_predict_model.add_external_input("conv1_w");// predict.AddInput(output + "_w");
                  deploy_predict_model.add_external_input("conv1_b");//predict.AddInput(output + "_b");


                  //   predict.AddConvOp(input, output + "_w", output + "_b", output, stride, padding, kernel)
                   { 
              
                    auto op = deploy_predict_model.add_op();// 
                    op->set_type("Conv");
                    // "conv1_b".size() ? std::vector<std::string>({"data", "conv1_w", "conv1_b"}): std::vector<std::string>({"data",, "conv1_w"});
                    for (auto input :{"data", "conv1_w","conv1_b"}) {//{"data", "conv1_w", "conv1_b"}
                        op->add_input(input);
                      } 
                    op->add_output("conv1");

                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("stride");        
                    arg->set_i(1);
                    auto arg1 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg2->set_name("kernel");        
                    arg2->set_i(5);
                    auto arg3 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg3->set_name("order");        
                    arg3->set_s("NCHW");
                  }
        }
    
        // model.predict.AddMaxPoolOp("conv1", "pool1", 2, 0, 2);
        {      

                    auto op = deploy_predict_model.add_op();// 
                    op->set_type("MaxPool");
                    op->add_input("conv1");
                    op->add_output("pool1");

                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("stride");        
                    arg->set_i(2);
                    auto arg1 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg2->set_name("kernel");        
                    arg2->set_i(2);
                    // if (group != 0) 
                    auto arg3 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg3->set_name("order");        
                    arg3->set_s("NCHW");
                    auto arg4 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg4->set_name("legacy_pad");        
                    arg4->set_i(3);
        }



        // void ModelUtil::AddConvOps(const std::string &input, const std::string &output,int in_size, int out_size, int stride, int padding,int kernel, bool test)
        //   model.AddConvOps("pool1", "conv2", 20, 50, 1, 0, 5, test);   

        {              
                  deploy_predict_model.add_external_input("conv2_w");// predict.AddInput(output + "_w");
                  deploy_predict_model.add_external_input("conv2_b");//predict.AddInput(output + "_b");
                  //   predict.AddConvOp(input, output + "_w", output + "_b", output, stride, padding, kernel)
                
                  { 
              
                    auto op = deploy_predict_model.add_op();// 
                    op->set_type("Conv");
                    for (auto input :{"pool1", "conv2_w", "conv2_b"}) {
                        op->add_input(input);
                      } 

                    op->add_output("conv2");
                    auto arg = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg->set_name("stride");        
                    arg->set_i(1);
                    auto arg1 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg2->set_name("kernel");        
                    arg2->set_i(5);
                    auto arg4 = op->add_arg();  //    net_add_arg(*op, "to", type);
                    arg4->set_name("order");        
                    arg4->set_s("NCHW");
                  }
        }


        // model.predict.AddMaxPoolOp("conv2", "pool2", 2, 0, 2);
        {      

                    auto op = deploy_predict_model.add_op();// 
                    op->set_type("MaxPool");
                 
                    op->add_input("conv2");
                    op->add_output("pool2");

                    auto arg = op->add_arg();  
                    arg->set_name("stride");        
                    arg->set_i(2);
                    auto arg1 = op->add_arg();  
                    arg1->set_name("pad");        
                    arg1->set_i(0);
                    auto arg2 = op->add_arg(); 
                    arg2->set_name("kernel");        
                    arg2->set_i(2);
                    auto arg3 = op->add_arg(); 
                    arg3->set_name("order");        
                    arg3->set_s("NCHW");
                    auto arg4 = op->add_arg();  
                    arg4->set_name("legacy_pad");        
                    arg4->set_i(3);
        }


      // model.AddFcOps("pool2", "fc3", 800, 500, test);
      {      
                              
                  deploy_predict_model.add_external_input("fc3_w");// predict.AddInput(output + "_w");
                  deploy_predict_model.add_external_input("fc3_b");//predict.AddInput(output + "_b");

                    // auto op = AddOp("FC", {input, w, b}, {output});
                  { 
              
                    auto op = deploy_predict_model.add_op();// 
                    op->set_type("FC");
                    for (auto input : {"pool2", "fc3_w", "fc3_b"}) {
                      op->add_input(input);
                    }
                    op->add_output("fc3");

                  }
        }

        //   // >>> fc3 = brew.relu(model, fc3, fc3)
        //   model.predict.AddReluOp("fc3", "fc3");
        {
                    auto op = deploy_predict_model.add_op();// 
                    op->set_type("Relu");
                    op->add_input("fc3");
                    op->add_output("fc3");

        }
        //   model.AddFcOps("fc3", "pred", 500, 10, test); 
        {       
                 
                  deploy_predict_model.add_external_input("pred_w");// predict.AddInput(output + "_w");
                  deploy_predict_model.add_external_input("pred_b");//predict.AddInput(output + "_b");

                  //   predict.AddFcOp(input, output + "_w", output + "_b", output);
                  // OperatorDef* NetUtil::AddFcOp(const std::string& input, const std::string& w,  const std::string& b, const std::string& output, int axis)
                  { 
              
                    auto op = deploy_predict_model.add_op();// 
                    op->set_type("FC");
                    for (auto input : {"fc3", "pred_w", "pred_b"}) {
                      op->add_input(input);
                    }
                    op->add_output("pred");       
                  }
        }

          // model.predict.AddSoftmaxOp("pred", "softmax");
      
        {          
                    auto op = deploy_predict_model.add_op();// 
                    op->set_type("Softmax");
                    op->add_input("pred");
                    op->add_output("softmax");              
        }
  }

  #ifdef WITH_CUDA
    if (!FLAGS_force_cpu) {
      train.SetDeviceCUDA();
      test.SetDeviceCUDA();
    }
  #endif

    std::cout << std::endl;

    // >>> workspace.RunNetOnce(train_model.param_init_net)
    CAFFE_ENFORCE(workspace.RunNetOnce(train_init_model));   //train_init_model   train_predict_model
    //  CAFFE_ENFORCE(workspace.RunNetOnce(train.init.net));   //train_init_model   train_predict_model

    // >>> workspace.CreateNet(train_model.net)
    CAFFE_ENFORCE(workspace.CreateNet(train_predict_model));
    //  CAFFE_ENFORCE(workspace.CreateNet(train.predict.net));

    std::cout << "training.." << std::endl;

    // >>> for i in range(total_iters):
    for (auto i = 1; i <= FLAGS_iters; i++) {
      // >>> workspace.RunNet(train_model.net.Proto().name)
      CAFFE_ENFORCE(workspace.RunNet(train_predict_model.name()));

      if (i % 10 == 0) {
      float w = workspace.GetBlob("accuracy")->Get<TensorCPU>().data<float>()[0];
      // float b = workspace.GetBlob("loss")->Get<TensorCPU>().data<float>()[0];
      float loss = workspace.GetBlob("loss")->Get<TensorCPU>().data<float>()[0];
      std::cout << "step: " << i << " loss: " << loss
                << " accuracy: " << w << std::endl;
      }
    }

    std::cout << std::endl;

    // >>> workspace.RunNetOnce(test_model.param_init_net)
    CAFFE_ENFORCE(workspace.RunNetOnce(test_init_model));//test_init_model   test_predict_model

    // >>> workspace.CreateNet(test_model.net)
    CAFFE_ENFORCE(workspace.CreateNet(test_predict_model));

    std::cout << "testing.." << std::endl;

    // >>> for i in range(100):
    for (auto i = 1; i <= FLAGS_test_runs; i++) {
      // >>> workspace.RunNet(test_model.net.Proto().name)
      CAFFE_ENFORCE(workspace.RunNet(test_predict_model.name()));
      if (i % 10 == 0) {
      float w = workspace.GetBlob("accuracy")->Get<TensorCPU>().data<float>()[0];
      float loss = workspace.GetBlob("loss")->Get<TensorCPU>().data<float>()[0];
      std::cout << "step: " << i  << " accuracy: " << w << std::endl;
      }




    }

    // with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
    // fid.write(str(deploy_model.net.Proto()))
    for (auto &param : deploy_predict_model.external_input()) {   // deploy_predict_model.
      // auto tensor = BlobUtil(*workspace.GetBlob(param)).Get();

      auto tensor = workspace.GetBlob(param)->Get<TensorCPU>();
      auto op =deploy_init_model.add_op();
      op->set_type("GivenTensorFill");
      auto arg1 = op->add_arg();
      arg1->set_name("shape");
      for (auto d : tensor.dims()) {
        arg1->add_ints(d);
      }
      auto arg2 = op->add_arg();
      arg2->set_name("values");
      auto data = tensor.data<float>();
      for (auto i = 0; i < tensor.size(); i++) {
        arg2->add_floats(data[i]);
      }
      op->add_output(param);
    }

    std::cout << std::endl;
    std::cout << "saving model.. (tmp/mnist_%_net.pb)" << std::endl;

    // deploy_predict_model.WriteText("tmp/mnist_predict_net.pbtxt");
     WriteProtoToBinaryFile(deploy_predict_model, "tmp/mnist_predict_net.pbtxt");

    // deploy.Write("tmp/mnist");

    Write_init_net(deploy_init_model, "tmp/mnist") ; 
    Write_predict_net(deploy_predict_model, "tmp/mnist") ;
   WriteText(deploy_init_model, "tmp/mnist") ; 
     WriteText(deploy_predict_model, "tmp/mnist") ; 

  }


  void predict_example() {
    std::vector<float> data_for_2(
        {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0.2, 0.6, 0.8, 0.9, 0.7,
        0.2, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0.8, 0.3, 0.2, 0.2, 0.7,
        0.9, 0.4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0.3, 0,   0,   0,   0,
        0.4, 0.9, 0.3, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0.4, 0.8, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0.8, 0.6, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0.2, 0.8, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0.1, 0.9, 0.3, 0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0.7, 0.6, 0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0.8, 0.6, 0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0.9, 0.6, 0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0.2, 0.9, 0.3, 0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0.2, 0.9, 0.1, 0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0.2, 0.3, 0.3, 0,
        0,   0,   0.6, 0.7, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0.6, 0.9, 0.9, 0.9, 0.9,
        0.4, 0.2, 0.9, 0.3, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0.7, 0.8, 0.1, 0,   0,   0.4,
        0.9, 0.9, 0.8, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0.1, 0.9, 0.4, 0,   0,   0,   0,
        0.3, 0.9, 0.8, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0.2, 0.9, 0.1, 0,   0,   0,   0.3,
        0.9, 0.8, 0.8, 0.7, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0.1, 0.9, 0.1, 0,   0.2, 0.3, 0.9,
        0.8, 0.1, 0.1, 0.8, 0.7, 0.2, 0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0.7, 0.9, 0.8, 0.9, 0.9, 0.6,
        0.1, 0,   0,   0.1, 0.5, 0.9, 0.7, 0.2, 0.1, 0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0.5, 0.6, 0.3, 0,   0,
        0,   0,   0,   0,   0,   0.3, 0.8, 0.9, 0.2, 0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0});

    std::cout << "classifying image of decimal:";
    auto i = 0;
    for (auto d : data_for_2) {
      if (i % 28 == 0) std::cout << std::endl;
      std::cout << (d > 0 ? "[]" : "  ");
      i++;
    }
    std::cout << std::endl;

    #ifdef WITH_CUDA
      DeviceOption option;
      option.set_device_type(CUDA);
      new CUDAContext(option);
    #endif

      // setup perdictor
      NetDef init_model, predict_model;
      CAFFE_ENFORCE(ReadProtoFromFile("tmp/mnist_init_net.pb", &init_model));
      CAFFE_ENFORCE(ReadProtoFromFile("tmp/mnist_predict_net.pb", &predict_model));

    #ifdef WITH_CUDA
      init_model.mutable_device_option()->set_device_type(CUDA);
      predict_model.mutable_device_option()->set_device_type(CUDA);
    #endif

      // load parameters
      Workspace workspace("tmp");
      CAFFE_ENFORCE(workspace.RunNetOnce(init_model));

    // input image data for "2"
    #ifdef WITH_CUDA
      auto data = workspace.CreateBlob("data")->GetMutable<TensorCUDA>();
    #else
      auto data = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
    #endif
      TensorCPU input({1, 1, 28, 28}, data_for_2, NULL);
      data->CopyFrom(input);

      // run predictor
      CAFFE_ENFORCE(workspace.RunNetOnce(predict_model));

    // read prediction
    #ifdef WITH_CUDA
      auto softmax = TensorCPU(workspace.GetBlob("softmax")->Get<TensorCUDA>());
    #else
      auto softmax = workspace.GetBlob("softmax")->Get<TensorCPU>();
    #endif
      std::vector<float> probs(softmax.data<float>(),
                              softmax.data<float>() + softmax.size());
      auto max = std::max_element(probs.begin(), probs.end());
      auto label = std::distance(probs.begin(), max);
      std::cout << "predicted label: '" << label << "' with probability: " << *max
                << std::endl;
  }

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  caffe2::predict_example();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
