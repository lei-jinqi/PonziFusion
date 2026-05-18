pragma solidity ^0.4.24;

contract Ponzi {
    address public throne ;
    uint public price = 1 ether;

    function() payable{//fallback function
        if(msg.value < price) revert();
        throne.transfer(msg.value - 100 finney);//reward
        throne = msg.sender;//invest
        price *= 2;
    }
}